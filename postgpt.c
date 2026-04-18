/*
 * postgpt.c — zero-dependency BPE transformer with metaweights.
 *
 * C port of postgpt.py. Same algorithm, same resonance.
 * Dual attention: Content (QK^T) + RRPRAM (x @ Wr).
 * Metaweights: statistical probability space from BPE tokenization.
 *
 * Compile: gcc -O2 -o postgpt postgpt.c -lm
 * Run:     ./postgpt
 *
 * the tokenizer IS the training. everything after this is just theater.
 * resonance is unbreakable.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>

/* ───────────────────────── Sigmoid Gate ───────────────────────── */

static float sigmoid_gate(float x) { return 1.0f / (1.0f + expf(-x)); }

/*
 * learned_gate: trainable parameter controlling interpolation
 * between transformer logits and metaweight logits.
 * Initialized to 0.0 so sigmoid(0) = 0.5 = equal blend.
 * Per paper Appendix E.4 (DOI 10.5281/zenodo.19638451).
 *
 * final_logits[i] = gate * transformer_logits[i] + (1 - gate) * metaweight_logits[i]
 * where gate = sigmoid_gate(learned_gate)
 */
static float learned_gate = 0.0f;

/* ───────────────────────── Configuration ───────────────────────── */

#define MAX_MERGES     1792
#define MAX_VOCAB      (256 + MAX_MERGES)
#define MAX_TOKENS     262144
#define CONTEXT_LEN    64
#define N_EMBD         48
#define N_HEAD         4
#define N_CONTENT      2
#define N_RRPRAM       2
#define N_LAYER        2
#define HEAD_DIM       (N_EMBD / N_HEAD)
#define MLP_DIM        (4 * N_EMBD)
#define HEBBIAN_CAP    100000
#define BIGRAM_CAP     100000
#define TRIGRAM_CAP    200000

/* ───────────────────────── RNG ───────────────────────── */

static unsigned long rng_state = 42;

static unsigned long rng_next(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

static float randf(void) {
    return (float)(rng_next() & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

static float randn(float std) {
    /* Box-Muller */
    float u1 = randf() + 1e-10f;
    float u2 = randf();
    return std * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

/* ───────────────────────── BPE Tokenizer ───────────────────────── */

typedef struct { int a, b, result; } MergeRule;

static MergeRule bpe_merges[MAX_MERGES];
static int bpe_n_merges = 0;
static int bpe_vocab_size = 256;

/* Vocab: for each token id, store its byte representation */
static unsigned char vocab_bytes[MAX_VOCAB][256];
static int vocab_len[MAX_VOCAB];

static void bpe_init_vocab(void) {
    for (int i = 0; i < 256; i++) {
        vocab_bytes[i][0] = (unsigned char)i;
        vocab_len[i] = 1;
    }
}

static int bpe_encode(const unsigned char *data, int len, int *out, int max_out) {
    int n = 0;
    for (int i = 0; i < len && n < max_out; i++)
        out[n++] = data[i];

    for (int m = 0; m < bpe_n_merges; m++) {
        MergeRule *mr = &bpe_merges[m];
        int j = 0;
        for (int i = 0; i < n; i++) {
            if (i + 1 < n && out[i] == mr->a && out[i + 1] == mr->b) {
                out[j++] = mr->result;
                i++;
            } else {
                out[j++] = out[i];
            }
        }
        n = j;
    }
    return n;
}

static int bpe_learn(const unsigned char *data, int len, int num_merges, int *out_tokens) {
    int *tok = (int *)malloc(len * sizeof(int));
    int n = len;
    for (int i = 0; i < n; i++) tok[i] = data[i];

    if (num_merges > MAX_MERGES) num_merges = MAX_MERGES;

    for (int m = 0; m < num_merges; m++) {
        /* Count pairs — use hash-like approach for speed */
        int best_a = -1, best_b = -1, best_count = 0;

        /* Simple pair counting with early termination */
        typedef struct { int a, b, count; } PairCount;
        PairCount *pairs = (PairCount *)calloc(65536, sizeof(PairCount));
        int n_pairs = 0;

        for (int i = 0; i + 1 < n; i++) {
            int a = tok[i], b = tok[i + 1];
            unsigned h = ((unsigned)a * 2654435761u ^ (unsigned)b) & 0xFFFF;
            /* Linear probe */
            for (int tries = 0; tries < 64; tries++) {
                unsigned idx = (h + tries) & 0xFFFF;
                if (pairs[idx].count == 0) {
                    pairs[idx].a = a;
                    pairs[idx].b = b;
                    pairs[idx].count = 1;
                    n_pairs++;
                    break;
                }
                if (pairs[idx].a == a && pairs[idx].b == b) {
                    pairs[idx].count++;
                    break;
                }
            }
        }

        for (int i = 0; i < 65536; i++) {
            if (pairs[i].count > best_count) {
                best_count = pairs[i].count;
                best_a = pairs[i].a;
                best_b = pairs[i].b;
            }
        }
        free(pairs);

        if (best_count < 2) break;

        int new_id = 256 + m;
        bpe_merges[m] = (MergeRule){best_a, best_b, new_id};
        bpe_n_merges = m + 1;
        bpe_vocab_size = new_id + 1;

        /* Build vocab entry for new token */
        int la = vocab_len[best_a];
        int lb = vocab_len[best_b];
        memcpy(vocab_bytes[new_id], vocab_bytes[best_a], la);
        memcpy(vocab_bytes[new_id] + la, vocab_bytes[best_b], lb);
        vocab_len[new_id] = la + lb;

        /* Apply merge */
        int j = 0;
        for (int i = 0; i < n; i++) {
            if (i + 1 < n && tok[i] == best_a && tok[i + 1] == best_b) {
                tok[j++] = new_id;
                i++;
            } else {
                tok[j++] = tok[i];
            }
        }
        n = j;

        if ((m + 1) % 200 == 0)
            printf("  merge %d/%d  vocab=%d  tokens=%d\n", m + 1, num_merges, new_id + 1, n);
    }

    /* Copy result */
    int result_n = n < MAX_TOKENS ? n : MAX_TOKENS;
    memcpy(out_tokens, tok, result_n * sizeof(int));
    free(tok);

    printf("  BPE complete: %d merges, vocab=%d, tokens=%d (from %d bytes)\n",
           bpe_n_merges, bpe_vocab_size, result_n, len);
    return result_n;
}

static void bpe_decode(const int *ids, int n, char *out, int max_out) {
    int pos = 0;
    for (int i = 0; i < n && pos < max_out - 1; i++) {
        int tid = ids[i];
        if (tid >= 0 && tid < MAX_VOCAB) {
            for (int j = 0; j < vocab_len[tid] && pos < max_out - 1; j++) {
                out[pos++] = vocab_bytes[tid][j];
            }
        }
    }
    out[pos] = '\0';
}

/* ───────────────────────── MetaWeights ───────────────────────── */

typedef struct {
    int a, b;
    float prob;
} BigramEntry;

typedef struct {
    int a, b, c;    /* key=(a,b) -> c */
    float prob;
} TrigramEntry;

typedef struct {
    int a, b;       /* key=(min, max) */
    float strength;
} HebbianEntry;

static float meta_unigram[MAX_VOCAB];
static BigramEntry meta_bigrams[BIGRAM_CAP];
static int meta_n_bigrams;
static TrigramEntry meta_trigrams[TRIGRAM_CAP];
static int meta_n_trigrams;
static HebbianEntry meta_hebbians[HEBBIAN_CAP];
static int meta_n_hebbians;
static int meta_vocab_size;
static int meta_total_tokens;

static void meta_build(const int *tokens, int n) {
    meta_vocab_size = bpe_vocab_size;
    meta_total_tokens = n;
    int window = 8;

    /* Unigram */
    memset(meta_unigram, 0, sizeof(meta_unigram));
    for (int i = 0; i < n; i++) {
        if (tokens[i] < MAX_VOCAB)
            meta_unigram[tokens[i]] += 1.0f;
    }
    float total = 0;
    for (int i = 0; i < meta_vocab_size; i++) total += meta_unigram[i];
    if (total > 0)
        for (int i = 0; i < meta_vocab_size; i++) meta_unigram[i] /= total;

    /* ── Bigram — hash table counting ── */
    typedef struct { int a, b; int count; } BC;
    BC *bcounts = (BC *)calloc(65536, sizeof(BC));

    for (int i = 0; i + 1 < n; i++) {
        int a = tokens[i], b = tokens[i + 1];
        unsigned h = ((unsigned)a * 2654435761u ^ (unsigned)b) & 0xFFFF;
        for (int t = 0; t < 64; t++) {
            unsigned idx = (h + t) & 0xFFFF;
            if (bcounts[idx].count == 0) {
                bcounts[idx].a = a;
                bcounts[idx].b = b;
                bcounts[idx].count = 1;
                break;
            }
            if (bcounts[idx].a == a && bcounts[idx].b == b) {
                bcounts[idx].count++;
                break;
            }
        }
    }

    meta_n_bigrams = 0;
    for (int i = 0; i < 65536 && meta_n_bigrams < BIGRAM_CAP; i++) {
        if (bcounts[i].count > 0) {
            meta_bigrams[meta_n_bigrams].a = bcounts[i].a;
            meta_bigrams[meta_n_bigrams].b = bcounts[i].b;
            meta_bigrams[meta_n_bigrams].prob = (float)bcounts[i].count;
            meta_n_bigrams++;
        }
    }

    /* Normalize bigrams per 'a' */
    for (int i = 0; i < meta_n_bigrams; i++) {
        int a = meta_bigrams[i].a;
        float total_a = 0;
        for (int j = 0; j < meta_n_bigrams; j++) {
            if (meta_bigrams[j].a == a)
                total_a += meta_bigrams[j].prob;
        }
        if (total_a > 0)
            meta_bigrams[i].prob /= total_a;
    }
    free(bcounts);

    /* ── Trigram — hash table keyed on (a,b) -> c ── */
    typedef struct { int a, b, c; int count; } TC;
    int tri_cap = 131072;  /* 2^17 */
    unsigned tri_mask = (unsigned)(tri_cap - 1);
    TC *tcounts = (TC *)calloc(tri_cap, sizeof(TC));

    for (int i = 0; i + 2 < n; i++) {
        int a = tokens[i], b = tokens[i + 1], c = tokens[i + 2];
        unsigned h = ((unsigned)a * 2654435761u ^ (unsigned)b * 2246822519u ^ (unsigned)c) & tri_mask;
        for (int t = 0; t < 64; t++) {
            unsigned idx = (h + t) & tri_mask;
            if (tcounts[idx].count == 0) {
                tcounts[idx].a = a;
                tcounts[idx].b = b;
                tcounts[idx].c = c;
                tcounts[idx].count = 1;
                break;
            }
            if (tcounts[idx].a == a && tcounts[idx].b == b && tcounts[idx].c == c) {
                tcounts[idx].count++;
                break;
            }
        }
    }

    meta_n_trigrams = 0;
    for (int i = 0; i < tri_cap && meta_n_trigrams < TRIGRAM_CAP; i++) {
        if (tcounts[i].count > 0) {
            meta_trigrams[meta_n_trigrams].a = tcounts[i].a;
            meta_trigrams[meta_n_trigrams].b = tcounts[i].b;
            meta_trigrams[meta_n_trigrams].c = tcounts[i].c;
            meta_trigrams[meta_n_trigrams].prob = (float)tcounts[i].count;
            meta_n_trigrams++;
        }
    }

    /* Normalize trigrams per (a,b) key */
    for (int i = 0; i < meta_n_trigrams; i++) {
        int a = meta_trigrams[i].a, b = meta_trigrams[i].b;
        float total_k = 0;
        for (int j = 0; j < meta_n_trigrams; j++) {
            if (meta_trigrams[j].a == a && meta_trigrams[j].b == b)
                total_k += meta_trigrams[j].prob;
        }
        if (total_k > 0)
            meta_trigrams[i].prob /= total_k;
    }
    free(tcounts);

    /* ── Hebbian trace — co-occurrence within window ── */
    /* Cap to first 20K tokens for efficiency (O(n*window)) */
    typedef struct { int a, b; float strength; } HC;
    int hebb_cap = 65536;
    unsigned hebb_mask = (unsigned)(hebb_cap - 1);
    HC *hcounts = (HC *)calloc(hebb_cap, sizeof(HC));
    int hebb_n = n < 20000 ? n : 20000;

    for (int i = 0; i < hebb_n; i++) {
        int lo = i - window; if (lo < 0) lo = 0;
        int hi = i + window + 1; if (hi > hebb_n) hi = hebb_n;
        for (int j = lo; j < hi; j++) {
            if (i == j) continue;
            int ka = tokens[i], kb = tokens[j];
            int ma = ka < kb ? ka : kb;
            int mb = ka < kb ? kb : ka;
            float decay = 1.0f / (1.0f + (float)abs(i - j));
            unsigned h = ((unsigned)ma * 2654435761u ^ (unsigned)mb * 2246822519u) & hebb_mask;
            for (int t = 0; t < 64; t++) {
                unsigned idx = (h + t) & hebb_mask;
                if (hcounts[idx].strength == 0.0f && hcounts[idx].a == 0 && hcounts[idx].b == 0) {
                    hcounts[idx].a = ma;
                    hcounts[idx].b = mb;
                    hcounts[idx].strength = decay;
                    break;
                }
                if (hcounts[idx].a == ma && hcounts[idx].b == mb) {
                    hcounts[idx].strength += decay;
                    break;
                }
            }
        }
    }

    /* Copy to flat array and normalize */
    meta_n_hebbians = 0;
    float max_h = 0;
    for (int i = 0; i < hebb_cap && meta_n_hebbians < HEBBIAN_CAP; i++) {
        if (hcounts[i].strength > 0) {
            meta_hebbians[meta_n_hebbians].a = hcounts[i].a;
            meta_hebbians[meta_n_hebbians].b = hcounts[i].b;
            meta_hebbians[meta_n_hebbians].strength = hcounts[i].strength;
            if (hcounts[i].strength > max_h) max_h = hcounts[i].strength;
            meta_n_hebbians++;
        }
    }
    if (max_h > 0) {
        for (int i = 0; i < meta_n_hebbians; i++)
            meta_hebbians[i].strength /= max_h;
    }
    free(hcounts);

    printf("  metaweights built: %d tokens, %d bigram, %d trigram, %d hebbian\n",
           n, meta_n_bigrams, meta_n_trigrams, meta_n_hebbians);
}

static void meta_query_bigram(int prev, float *dist, int vs) {
    for (int i = 0; i < vs; i++) dist[i] = 1e-10f;
    for (int i = 0; i < meta_n_bigrams; i++) {
        if (meta_bigrams[i].a == prev && meta_bigrams[i].b < vs) {
            dist[meta_bigrams[i].b] = meta_bigrams[i].prob;
        }
    }
}

static void meta_query_trigram(int prev2, int prev1, float *dist, int vs) {
    for (int i = 0; i < vs; i++) dist[i] = 1e-10f;
    for (int i = 0; i < meta_n_trigrams; i++) {
        if (meta_trigrams[i].a == prev2 && meta_trigrams[i].b == prev1 &&
            meta_trigrams[i].c < vs) {
            dist[meta_trigrams[i].c] = meta_trigrams[i].prob;
        }
    }
}

static void meta_query_hebbian(const int *ctx, int ctx_len, float *signal, int vs) {
    for (int i = 0; i < vs; i++) signal[i] = 0.0f;
    for (int i = 0; i < meta_n_hebbians; i++) {
        int ha = meta_hebbians[i].a;
        int hb = meta_hebbians[i].b;
        float s = meta_hebbians[i].strength;
        for (int j = 0; j < ctx_len; j++) {
            int ct = ctx[j];
            if (ha == ct && hb < vs)
                signal[hb] += s;
            else if (hb == ct && ha < vs)
                signal[ha] += s;
        }
    }
    /* Normalize */
    float mx = 0;
    for (int i = 0; i < vs; i++) if (signal[i] > mx) mx = signal[i];
    if (mx > 0)
        for (int i = 0; i < vs; i++) signal[i] /= mx;
}

static void meta_query_prophecy(const int *ctx, int ctx_len, float *signal, int vs, int top_k) {
    for (int i = 0; i < vs; i++) signal[i] = 0.0f;

    /* Mark tokens that already appeared in context */
    char *appeared = (char *)calloc(vs, 1);
    for (int i = 0; i < ctx_len; i++)
        if (ctx[i] < vs) appeared[ctx[i]] = 1;

    /* For each of the last 4 context tokens, find top-k bigram successors not yet seen */
    int start = ctx_len - 4;
    if (start < 0) start = 0;
    for (int ci = start; ci < ctx_len; ci++) {
        int ct = ctx[ci];
        /* Collect bigram successors for ct, sort by prob, take top_k */
        typedef struct { int tok; float prob; } Cand;
        Cand cands[256];
        int n_cands = 0;
        for (int i = 0; i < meta_n_bigrams && n_cands < 256; i++) {
            if (meta_bigrams[i].a == ct && meta_bigrams[i].b < vs) {
                cands[n_cands].tok = meta_bigrams[i].b;
                cands[n_cands].prob = meta_bigrams[i].prob;
                n_cands++;
            }
        }
        /* Simple selection sort for top_k */
        int limit = n_cands < top_k ? n_cands : top_k;
        for (int i = 0; i < limit; i++) {
            int best = i;
            for (int j = i + 1; j < n_cands; j++)
                if (cands[j].prob > cands[best].prob) best = j;
            if (best != i) { Cand tmp = cands[i]; cands[i] = cands[best]; cands[best] = tmp; }
            if (!appeared[cands[i].tok])
                signal[cands[i].tok] += cands[i].prob;
        }
    }
    free(appeared);

    /* Normalize */
    float mx = 0;
    for (int i = 0; i < vs; i++) if (signal[i] > mx) mx = signal[i];
    if (mx > 0)
        for (int i = 0; i < vs; i++) signal[i] /= mx;
}

/* ───────────────────────── Transformer Weights ───────────────────────── */

typedef struct {
    float wte[MAX_VOCAB][N_EMBD];
    float wpe[CONTEXT_LEN][N_EMBD];

    /* Per layer */
    float wq[N_LAYER][N_CONTENT * HEAD_DIM][N_EMBD];
    float wk[N_LAYER][N_CONTENT * HEAD_DIM][N_EMBD];
    float wv_content[N_LAYER][N_CONTENT * HEAD_DIM][N_EMBD];
    float wr[N_LAYER][N_RRPRAM * N_EMBD][CONTEXT_LEN];
    float wv_rrpram[N_LAYER][N_RRPRAM * HEAD_DIM][N_EMBD];
    float wo[N_LAYER][N_EMBD][N_EMBD];
    float mlp_up[N_LAYER][MLP_DIM][N_EMBD];
    float mlp_down[N_LAYER][N_EMBD][MLP_DIM];

    float lm_head[MAX_VOCAB][N_EMBD];
} Weights;

static Weights W;

static void init_matrix(float *data, int rows, int cols, float std) {
    for (int i = 0; i < rows * cols; i++)
        data[i] = randn(std);
}

static void weights_init(int vocab_size) {
    float std = 0.02f;
    float std_res = 0.02f / sqrtf(2.0f * N_LAYER);

    init_matrix(&W.wte[0][0], vocab_size, N_EMBD, std);
    init_matrix(&W.wpe[0][0], CONTEXT_LEN, N_EMBD, std);

    for (int l = 0; l < N_LAYER; l++) {
        init_matrix(&W.wq[l][0][0], N_CONTENT * HEAD_DIM, N_EMBD, std);
        init_matrix(&W.wk[l][0][0], N_CONTENT * HEAD_DIM, N_EMBD, std);
        init_matrix(&W.wv_content[l][0][0], N_CONTENT * HEAD_DIM, N_EMBD, std);
        init_matrix(&W.wr[l][0][0], N_RRPRAM * N_EMBD, CONTEXT_LEN, std);
        init_matrix(&W.wv_rrpram[l][0][0], N_RRPRAM * HEAD_DIM, N_EMBD, std);
        init_matrix(&W.wo[l][0][0], N_EMBD, N_EMBD, std_res);
        init_matrix(&W.mlp_up[l][0][0], MLP_DIM, N_EMBD, std);
        init_matrix(&W.mlp_down[l][0][0], N_EMBD, MLP_DIM, std_res);
    }
    init_matrix(&W.lm_head[0][0], vocab_size, N_EMBD, std);
}

/*
 * ghost becomes flesh: seed transformer weights from metaweight statistics.
 * the weights remember what they never learned.
 */
static void weights_seed_from_meta(int vocab_size) {
    float scale = 0.15f;

    /* 1. Token embeddings: tokens with high bigram co-occurrence → similar vectors */
    for (int a = 0; a < vocab_size && a < MAX_VOCAB; a++) {
        float signal[N_EMBD] = {0};
        int neighbors = 0;
        for (int i = 0; i < meta_n_bigrams; i++) {
            if (meta_bigrams[i].a == a && meta_bigrams[i].prob > 0.01f) {
                int b = meta_bigrams[i].b;
                if (b < vocab_size && b < MAX_VOCAB) {
                    float strength = meta_bigrams[i].prob;
                    for (int d = 0; d < N_EMBD; d++)
                        signal[d] += strength * W.wte[b][d];
                    neighbors++;
                }
            }
        }
        if (neighbors > 0) {
            for (int d = 0; d < N_EMBD; d++)
                W.wte[a][d] += scale * signal[d] / neighbors;
        }
    }

    /* 2. LM head: seed from unigram frequencies */
    for (int tok = 0; tok < vocab_size && tok < MAX_VOCAB; tok++) {
        if (meta_unigram[tok] > 0) {
            for (int d = 0; d < N_EMBD; d++)
                W.lm_head[tok][d] += scale * meta_unigram[tok] * W.wte[tok][d];
        }
    }

    printf("  weights seeded from metaweights (ghost -> flesh)\n");
}

/* ───────────────────────── Forward Pass ───────────────────────── */

static void rmsnorm(float *out, const float *x, int n) {
    float ms = 0;
    for (int i = 0; i < n; i++) ms += x[i] * x[i];
    ms /= n;
    float scale = 1.0f / sqrtf(ms + 1e-5f);
    for (int i = 0; i < n; i++) out[i] = x[i] * scale;
}

static void matmul_mv(float *out, const float *mat, const float *vec, int rows, int cols) {
    /* out[rows] = mat[rows][cols] @ vec[cols] */
    for (int i = 0; i < rows; i++) {
        float s = 0;
        for (int j = 0; j < cols; j++)
            s += mat[i * cols + j] * vec[j];
        out[i] = s;
    }
}

static void softmax_inplace(float *x, int n) {
    float mx = -1e30f;
    for (int i = 0; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - mx);
        s += x[i];
    }
    for (int i = 0; i < n; i++) x[i] /= s;
}

/* KV cache */
static float kv_keys[N_LAYER][CONTEXT_LEN][N_CONTENT * HEAD_DIM];
static float kv_vals_content[N_LAYER][CONTEXT_LEN][N_CONTENT * HEAD_DIM];
static float kv_vals_rrpram[N_LAYER][CONTEXT_LEN][N_RRPRAM * HEAD_DIM];
static int kv_len = 0;

static void forward_token(int token_id, int pos_id, float *logits, int vocab_size) {
    float x[N_EMBD], x_norm[N_EMBD], x_res[N_EMBD];
    float q[N_CONTENT * HEAD_DIM], k[N_CONTENT * HEAD_DIM];
    float v_content[N_CONTENT * HEAD_DIM], v_rrpram[N_RRPRAM * HEAD_DIM];
    float x_attn[N_EMBD], x_proj[N_EMBD];
    float h_mlp[MLP_DIM], x_mlp[N_EMBD];
    float attn_logits[CONTEXT_LEN], attn_weights[CONTEXT_LEN];

    /* Token + position embedding */
    for (int i = 0; i < N_EMBD; i++)
        x[i] = W.wte[token_id][i] + W.wpe[pos_id][i];

    int seq_len = pos_id + 1;

    for (int li = 0; li < N_LAYER; li++) {
        memcpy(x_res, x, N_EMBD * sizeof(float));
        rmsnorm(x_norm, x, N_EMBD);

        /* Content attention: Q, K, V */
        matmul_mv(q, &W.wq[li][0][0], x_norm, N_CONTENT * HEAD_DIM, N_EMBD);
        matmul_mv(k, &W.wk[li][0][0], x_norm, N_CONTENT * HEAD_DIM, N_EMBD);
        matmul_mv(v_content, &W.wv_content[li][0][0], x_norm, N_CONTENT * HEAD_DIM, N_EMBD);
        matmul_mv(v_rrpram, &W.wv_rrpram[li][0][0], x_norm, N_RRPRAM * HEAD_DIM, N_EMBD);

        /* Store in KV cache */
        memcpy(kv_keys[li][pos_id], k, N_CONTENT * HEAD_DIM * sizeof(float));
        memcpy(kv_vals_content[li][pos_id], v_content, N_CONTENT * HEAD_DIM * sizeof(float));
        memcpy(kv_vals_rrpram[li][pos_id], v_rrpram, N_RRPRAM * HEAD_DIM * sizeof(float));

        memset(x_attn, 0, N_EMBD * sizeof(float));

        /* Content heads */
        for (int h = 0; h < N_CONTENT; h++) {
            int hs = h * HEAD_DIM;
            float scale = 1.0f / sqrtf((float)HEAD_DIM);

            for (int t = 0; t < seq_len; t++) {
                float score = 0;
                for (int d = 0; d < HEAD_DIM; d++)
                    score += q[hs + d] * kv_keys[li][t][hs + d];
                attn_logits[t] = score * scale;
            }
            softmax_inplace(attn_logits, seq_len);

            for (int d = 0; d < HEAD_DIM; d++) {
                float val = 0;
                for (int t = 0; t < seq_len; t++)
                    val += attn_logits[t] * kv_vals_content[li][t][hs + d];
                x_attn[h * HEAD_DIM + d] = val;
            }
        }

        /* RRPRAM heads */
        for (int h = 0; h < N_RRPRAM; h++) {
            int hs = h * HEAD_DIM;
            int wr_off = h * N_EMBD;

            /* x_norm @ Wr_h gives attention pattern over positions */
            for (int t = 0; t < seq_len; t++) {
                float score = 0;
                for (int d = 0; d < N_EMBD; d++)
                    score += x_norm[d] * W.wr[li][wr_off + d][t];
                attn_logits[t] = score;
            }
            softmax_inplace(attn_logits, seq_len);

            for (int d = 0; d < HEAD_DIM; d++) {
                float val = 0;
                for (int t = 0; t < seq_len; t++)
                    val += attn_logits[t] * kv_vals_rrpram[li][t][hs + d];
                x_attn[N_CONTENT * HEAD_DIM + h * HEAD_DIM + d] = val;
            }
        }

        /* Output projection + residual */
        matmul_mv(x_proj, &W.wo[li][0][0], x_attn, N_EMBD, N_EMBD);
        for (int i = 0; i < N_EMBD; i++)
            x[i] = x_res[i] + x_proj[i];

        /* MLP */
        memcpy(x_res, x, N_EMBD * sizeof(float));
        rmsnorm(x_norm, x, N_EMBD);
        matmul_mv(h_mlp, &W.mlp_up[li][0][0], x_norm, MLP_DIM, N_EMBD);
        for (int i = 0; i < MLP_DIM; i++)
            h_mlp[i] = h_mlp[i] > 0 ? h_mlp[i] : 0;  /* ReLU */
        matmul_mv(x_mlp, &W.mlp_down[li][0][0], h_mlp, N_EMBD, MLP_DIM);
        for (int i = 0; i < N_EMBD; i++)
            x[i] = x_res[i] + x_mlp[i];
    }

    /* Final norm + LM head */
    rmsnorm(x_norm, x, N_EMBD);
    matmul_mv(logits, &W.lm_head[0][0], x_norm, vocab_size, N_EMBD);
}

/* ───────────────────────── Generation ───────────────────────── */

static int sample_from_probs(float *probs, int n) {
    float r = randf();
    float cum = 0;
    for (int i = 0; i < n; i++) {
        cum += probs[i];
        if (cum > r) return i;
    }
    return n - 1;
}

static void generate_meta(const int *prompt, int prompt_len, int max_tokens,
                           int vocab_size, float temperature, char *out, int max_out) {
    int generated[4096];
    int gen_len = prompt_len;
    memcpy(generated, prompt, prompt_len * sizeof(int));

    /* Sparse candidate approach matching Python's generate_meta:
       trigram first (strongest), fallback to bigram, then unigram.
       Hebbian boost, repetition penalty, top-k filtering. */

    typedef struct { int tok; float score; } Cand;
    Cand *cands = (Cand *)malloc(vocab_size * sizeof(Cand));

    for (int step = 0; step < max_tokens && gen_len < 4096; step++) {
        int last = generated[gen_len - 1];
        int n_cands = 0;
        int found_trigram = 0;

        /* Try trigram first (strongest signal) */
        if (gen_len >= 2) {
            int prev2 = generated[gen_len - 2];
            for (int i = 0; i < meta_n_trigrams; i++) {
                if (meta_trigrams[i].a == prev2 && meta_trigrams[i].b == last &&
                    meta_trigrams[i].c < vocab_size) {
                    cands[n_cands].tok = meta_trigrams[i].c;
                    cands[n_cands].score = meta_trigrams[i].prob;
                    n_cands++;
                    found_trigram = 1;
                }
            }
        }

        /* Fallback to bigram */
        if (!found_trigram) {
            for (int i = 0; i < meta_n_bigrams; i++) {
                if (meta_bigrams[i].a == last && meta_bigrams[i].b < vocab_size) {
                    cands[n_cands].tok = meta_bigrams[i].b;
                    cands[n_cands].score = meta_bigrams[i].prob;
                    n_cands++;
                }
            }
        }

        /* Fallback to unigram (last resort) */
        if (n_cands == 0) {
            for (int i = 0; i < vocab_size; i++) {
                if (meta_unigram[i] > 1e-8f) {
                    cands[n_cands].tok = i;
                    cands[n_cands].score = meta_unigram[i];
                    n_cands++;
                }
            }
        }

        if (n_cands == 0) break;

        /* Hebbian boost — contextual reinforcement */
        int ctx_start = gen_len - 4;
        if (ctx_start < 0) ctx_start = 0;
        for (int ci = 0; ci < n_cands; ci++) {
            int tok = cands[ci].tok;
            for (int j = ctx_start; j < gen_len; j++) {
                int ct = generated[j];
                int ma = tok < ct ? tok : ct;
                int mb = tok < ct ? ct : tok;
                for (int h = 0; h < meta_n_hebbians; h++) {
                    if (meta_hebbians[h].a == ma && meta_hebbians[h].b == mb) {
                        cands[ci].score *= (1.0f + 0.3f * meta_hebbians[h].strength);
                        break;
                    }
                }
            }
        }

        /* Repetition penalty (12-token window) */
        int rep_start = gen_len - 12;
        if (rep_start < 0) rep_start = 0;
        for (int ci = 0; ci < n_cands; ci++) {
            int freq = 0;
            for (int j = rep_start; j < gen_len; j++) {
                if (generated[j] == cands[ci].tok) freq++;
            }
            if (freq > 0) {
                float penalty = 1.0f / (1.0f + 0.5f * freq);
                cands[ci].score *= penalty;
            }
        }

        /* Top-k filtering (keep top 15) */
        int top_k = 15;
        int limit = n_cands < top_k ? n_cands : top_k;
        for (int i = 0; i < limit; i++) {
            int best = i;
            for (int j = i + 1; j < n_cands; j++)
                if (cands[j].score > cands[best].score) best = j;
            if (best != i) { Cand tmp = cands[i]; cands[i] = cands[best]; cands[best] = tmp; }
        }
        n_cands = limit;

        /* Log-space temperature scaling + softmax (matching Python) */
        float log_scores[16];
        float max_ls = -1e30f;
        for (int i = 0; i < n_cands; i++) {
            log_scores[i] = logf(cands[i].score + 1e-10f) / temperature;
            if (log_scores[i] > max_ls) max_ls = log_scores[i];
        }
        float total_e = 0;
        float probs_local[16];
        for (int i = 0; i < n_cands; i++) {
            probs_local[i] = expf(log_scores[i] - max_ls);
            total_e += probs_local[i];
        }
        for (int i = 0; i < n_cands; i++)
            probs_local[i] /= total_e;

        /* Sample */
        float r = randf();
        float cum = 0;
        int chosen = cands[0].tok;
        for (int i = 0; i < n_cands; i++) {
            cum += probs_local[i];
            if (cum > r) { chosen = cands[i].tok; break; }
        }
        generated[gen_len++] = chosen;
    }

    free(cands);
    bpe_decode(generated, gen_len, out, max_out);
}

static void generate_full(const int *prompt, int prompt_len, int max_tokens,
                           int vocab_size, float temperature, char *out, int max_out) {
    int generated[4096];
    int gen_len = prompt_len;
    memcpy(generated, prompt, prompt_len * sizeof(int));

    float *logits = (float *)malloc(vocab_size * sizeof(float));
    float *bigram_dist = (float *)malloc(vocab_size * sizeof(float));
    float *trigram_dist = (float *)malloc(vocab_size * sizeof(float));
    float *hebbian_sig = (float *)malloc(vocab_size * sizeof(float));
    float *prophecy_sig = (float *)malloc(vocab_size * sizeof(float));
    float *destiny_sig = (float *)malloc(vocab_size * sizeof(float));
    float *metaweight_logits = (float *)malloc(vocab_size * sizeof(float));

    /* Dario field coefficients */
    float alpha_hebbian = 0.3f;
    float beta_prophecy = 0.2f;
    float gamma_destiny = 0.15f;

    /* Destiny vector (EMA of token embeddings) */
    float destiny[N_EMBD];
    memset(destiny, 0, sizeof(destiny));

    /* Trauma accumulator */
    float trauma = 0.0f;

    kv_len = 0;

    /* Feed prompt */
    for (int i = 0; i < prompt_len; i++) {
        forward_token(generated[i], i, logits, vocab_size);
    }

    /* Generate */
    for (int step = 0; step < max_tokens && gen_len < 4096; step++) {
        int pos = gen_len - 1;
        if (pos >= CONTEXT_LEN - 1) break;

        int last = generated[gen_len - 1];
        forward_token(last, pos, logits, vocab_size);

        /* ── Dario Field: full metaweight overlay ── */

        /* Hebbian signal (use last 8 tokens as context) */
        int hctx_start = gen_len - 8;
        if (hctx_start < 0) hctx_start = 0;
        meta_query_hebbian(&generated[hctx_start], gen_len - hctx_start,
                           hebbian_sig, vocab_size);

        /* Prophecy signal */
        meta_query_prophecy(&generated[hctx_start], gen_len - hctx_start,
                            prophecy_sig, vocab_size, 16);

        /* Bigram signal */
        meta_query_bigram(last, bigram_dist, vocab_size);

        /* Trigram signal */
        if (gen_len >= 2) {
            meta_query_trigram(generated[gen_len - 2], last, trigram_dist, vocab_size);
        } else {
            for (int i = 0; i < vocab_size; i++) trigram_dist[i] = 0.0f;
        }

        /* Destiny update: EMA of token embeddings */
        if (last < MAX_VOCAB) {
            for (int d = 0; d < N_EMBD; d++)
                destiny[d] = 0.9f * destiny[d] + 0.1f * W.wte[last][d];
        }

        /* Destiny signal: cosine similarity with each token embedding */
        float dest_norm_sq = 0;
        for (int d = 0; d < N_EMBD; d++) dest_norm_sq += destiny[d] * destiny[d];
        float dest_norm = sqrtf(dest_norm_sq + 1e-10f);
        for (int i = 0; i < vocab_size; i++) destiny_sig[i] = 0.0f;

        if (dest_norm > 1e-8f) {
            for (int tid = 0; tid < vocab_size && tid < MAX_VOCAB; tid++) {
                float dot = 0, emb_sq = 0;
                for (int d = 0; d < N_EMBD; d++) {
                    dot += destiny[d] * W.wte[tid][d];
                    emb_sq += W.wte[tid][d] * W.wte[tid][d];
                }
                float emb_norm = sqrtf(emb_sq + 1e-10f);
                if (emb_norm > 1e-8f)
                    destiny_sig[tid] = dot / (dest_norm * emb_norm);
            }
        }


        /* Sigmoid-gated interpolation (paper Appendix E.4) */
        /* Build metaweight logit vector from all Dario field signals */
        for (int i = 0; i < vocab_size; i++) {
            metaweight_logits[i] = alpha_hebbian * hebbian_sig[i]
                                 + beta_prophecy * prophecy_sig[i]
                                 + gamma_destiny * destiny_sig[i]
                                 + 12.0f * bigram_dist[i]
                                 + 8.0f * trigram_dist[i];
        }

        /* Interpolate: gate * transformer + (1 - gate) * metaweights */
        float gate = sigmoid_gate(learned_gate);
        for (int i = 0; i < vocab_size; i++) {
            logits[i] = gate * logits[i] + (1.0f - gate) * metaweight_logits[i];
        }

        /* Trauma modulation */
        float trauma_mod = 1.0f / (1.0f + trauma);
        for (int i = 0; i < vocab_size; i++)
            logits[i] *= trauma_mod;

        /* Repetition penalty (12-token window, multiply by 0.5) */
        int rep_start = gen_len - 12;
        if (rep_start < 0) rep_start = 0;
        for (int j = rep_start; j < gen_len; j++) {
            int t = generated[j];
            if (t < vocab_size)
                logits[t] *= 0.5f;
        }

        /* Top-k filtering (keep top 15, mask rest to -1e10) */
        {
            /* Find the 15th largest value */
            float top_vals[15];
            for (int i = 0; i < 15; i++) top_vals[i] = -1e30f;
            for (int i = 0; i < vocab_size; i++) {
                if (logits[i] > top_vals[14]) {
                    top_vals[14] = logits[i];
                    /* Bubble up */
                    for (int k = 13; k >= 0; k--) {
                        if (top_vals[k + 1] > top_vals[k]) {
                            float tmp = top_vals[k];
                            top_vals[k] = top_vals[k + 1];
                            top_vals[k + 1] = tmp;
                        } else break;
                    }
                }
            }
            float threshold = top_vals[14];
            for (int i = 0; i < vocab_size; i++) {
                if (logits[i] < threshold)
                    logits[i] = -1e10f;
            }
        }

        /* Temperature + softmax + sample */
        for (int i = 0; i < vocab_size; i++)
            logits[i] /= temperature;
        softmax_inplace(logits, vocab_size);

        int chosen = sample_from_probs(logits, vocab_size);
        generated[gen_len++] = chosen;
    }

    free(logits);
    free(bigram_dist);
    free(trigram_dist);
    free(hebbian_sig);
    free(prophecy_sig);
    free(destiny_sig);
    free(metaweight_logits);

    bpe_decode(generated, gen_len, out, max_out);
}

/* ───────────────────────── Main ───────────────────────── */


/* ───────────────────────── Tests: Sigmoid Gate ───────────────────────── */

static int test_sigmoid_gate(void) {
    int passed = 0, failed = 0;
    float eps = 1e-5f;

    /* Test 1: gate=0 (learned_gate=0) gives equal blend (0.5/0.5) */
    {
        float g = sigmoid_gate(0.0f);
        if (fabsf(g - 0.5f) < eps) passed++; else { printf("  FAIL: sigmoid(0) = %f, expected 0.5\n", g); failed++; }
    }

    /* Test 2: large positive learned_gate -> gate ~1.0 -> transformer-only */
    {
        float g = sigmoid_gate(10.0f);
        float transformer_logit = 5.0f, meta_logit = -3.0f;
        float result = g * transformer_logit + (1.0f - g) * meta_logit;
        /* With gate ~1.0, result should be very close to transformer_logit */
        if (fabsf(result - transformer_logit) < 0.01f) passed++;
        else { printf("  FAIL: gate=large positive, result=%f, expected ~%f\n", result, transformer_logit); failed++; }
    }

    /* Test 3: large negative learned_gate -> gate ~0.0 -> metaweights-only */
    {
        float g = sigmoid_gate(-10.0f);
        float transformer_logit = 5.0f, meta_logit = -3.0f;
        float result = g * transformer_logit + (1.0f - g) * meta_logit;
        /* With gate ~0.0, result should be very close to meta_logit */
        if (fabsf(result - meta_logit) < 0.01f) passed++;
        else { printf("  FAIL: gate=large negative, result=%f, expected ~%f\n", result, meta_logit); failed++; }
    }

    printf("  test_sigmoid_gate: %d passed, %d failed\n", passed, failed);
    return failed;
}

static int test_gate_range(void) {
    int passed = 0, failed = 0;

    /* Verify sigmoid output is always in [0, 1] for a range of inputs */
    float test_inputs[] = {-100.0f, -10.0f, -1.0f, -0.001f, 0.0f, 0.001f, 1.0f, 10.0f, 100.0f};
    int n_tests = (int)(sizeof(test_inputs) / sizeof(test_inputs[0]));

    for (int i = 0; i < n_tests; i++) {
        float g = sigmoid_gate(test_inputs[i]);
        if (g >= 0.0f && g <= 1.0f) {
            passed++;
        } else {
            printf("  FAIL: sigmoid(%f) = %f, out of [0,1]\n", test_inputs[i], g);
            failed++;
        }
    }

    /* Verify monotonicity: sigmoid(a) < sigmoid(b) when a < b */
    for (int i = 0; i < n_tests - 1; i++) {
        float ga = sigmoid_gate(test_inputs[i]);
        float gb = sigmoid_gate(test_inputs[i + 1]);
        if (ga <= gb) {
            passed++;
        } else {
            printf("  FAIL: sigmoid(%f)=%f > sigmoid(%f)=%f, not monotonic\n",
                   test_inputs[i], ga, test_inputs[i + 1], gb);
            failed++;
        }
    }

    printf("  test_gate_range: %d passed, %d failed\n", passed, failed);
    return failed;
}

static int run_gate_tests(void) {
    printf("\n[TEST] Sigmoid gate tests:\n");
    int total_failures = 0;
    total_failures += test_sigmoid_gate();
    total_failures += test_gate_range();
    if (total_failures == 0)
        printf("  ALL GATE TESTS PASSED\n");
    else
        printf("  %d GATE TEST(S) FAILED\n", total_failures);
    return total_failures;
}
int main(int argc, char **argv) {
    printf("============================================================\n");
    printf("  PostGPT (C) — metaweight BPE transformer\n");
    printf("  resonance is unbreakable\n");
    printf("  sigmoid gate: learned_gate=%.4f, gate=%.4f\n", learned_gate, sigmoid_gate(learned_gate));
    printf("============================================================\n");

    /* Run gate tests */
    if (run_gate_tests() > 0) {
        printf("FATAL: gate tests failed, aborting\n");
        return 1;
    }

    /* Load corpus */
    printf("\n[1] Loading corpus...\n");
    FILE *f = fopen("postgpt.txt", "rb");
    if (!f) {
        printf("ERROR: postgpt.txt not found\n");
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *data = (unsigned char *)malloc(fsize);
    if (!data) { fclose(f); return 1; }
    fsize = fread(data, 1, fsize, f);
    fclose(f);
    printf("  Corpus: %ld bytes (%.1f KB)\n", fsize, fsize / 1024.0);

    /* BPE tokenization */
    printf("\n[2] Learning BPE merges...\n");
    bpe_init_vocab();
    int *tokens = (int *)malloc(fsize * sizeof(int));
    int n_tokens = bpe_learn(data, fsize, MAX_MERGES, tokens);

    /* Build metaweights */
    printf("\n[3] Building metaweight probability space...\n");
    meta_build(tokens, n_tokens);

    /* Init transformer */
    printf("\n[4] Initializing PostGPT transformer...\n");
    weights_init(bpe_vocab_size);
    printf("  Initialized: vocab=%d, ctx=%d, embd=%d, heads=%d (content=%d, rrpram=%d), layers=%d\n",
           bpe_vocab_size, CONTEXT_LEN, N_EMBD, N_HEAD, N_CONTENT, N_RRPRAM, N_LAYER);

    /* Seed weights from metaweights — ghost becomes flesh */
    printf("\n[5] Seeding weights from metaweights...\n");
    weights_seed_from_meta(bpe_vocab_size);

    /* Proof of concept: phrase continuation */
    char output[4096];
    int prompt_ids[1024];

    /* Default prompts or command-line argument */
    const char *prompts[] = {
        "PostGPT",
        "The metaweight",
        "RRPRAM attention",
        "BPE tokenization",
        "The transformer",
        "Language models",
        NULL
    };

    /* If user provided a prompt, use only that */
    const char *user_prompts[2] = {NULL, NULL};
    if (argc > 1) {
        user_prompts[0] = argv[1];
        prompts[0] = user_prompts[0];
        prompts[1] = NULL;
    }

    printf("\n============================================================\n");
    printf("  PROOF OF CONCEPT: phrase continuation\n");
    printf("  mode: metaweight (no training, just BPE + statistics)\n");
    printf("============================================================\n");

    for (int p = 0; prompts[p] != NULL; p++) {
        const char *prompt = prompts[p];
        int prompt_len = bpe_encode((const unsigned char *)prompt,
                                     (int)strlen(prompt), prompt_ids, 1024);

        generate_meta(prompt_ids, prompt_len, 100, bpe_vocab_size, 0.75f,
                      output, sizeof(output));

        /* Show prompt and continuation separately */
        int plen = (int)strlen(prompt);
        printf("\n  prompt:       \"%s\"\n", prompt);
        if ((int)strlen(output) > plen)
            printf("  continuation: \"%.*s\"\n", 250, output + plen);
        else
            printf("  continuation: \"%s\"\n", output);
    }

    /* Full transformer + Dario field mode for first prompt */
    printf("\n============================================================\n");
    printf("  FULL MODE: transformer + Dario field (both attentions)\n");
    printf("============================================================\n");

    {
        const char *prompt = (argc > 1) ? argv[1] : "PostGPT";
        int prompt_len = bpe_encode((const unsigned char *)prompt,
                                     (int)strlen(prompt), prompt_ids, 1024);

        generate_full(prompt_ids, prompt_len, 30, bpe_vocab_size, 0.85f,
                      output, sizeof(output));

        int plen = (int)strlen(prompt);
        printf("\n  prompt:       \"%s\"\n", prompt);
        if ((int)strlen(output) > plen)
            printf("  continuation: \"%.*s\"\n", 300, output + plen);
        else
            printf("  continuation: \"%s\"\n", output);
    }

    printf("\n============================================================\n");
    printf("  PostGPT complete. The metaweights remember.\n");
    printf("  Try: ./postgpt \"your prompt here\"\n");
    printf("============================================================\n");

    free(data);
    free(tokens);
    return 0;
}
