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
 * resonance is unbreakable.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>

/* ───────────────────────── Configuration ───────────────────────── */

#define MAX_MERGES     1024
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

static float meta_unigram[MAX_VOCAB];
static BigramEntry meta_bigrams[BIGRAM_CAP];
static int meta_n_bigrams;
static int meta_vocab_size;
static int meta_total_tokens;

static void meta_build(const int *tokens, int n) {
    meta_vocab_size = bpe_vocab_size;
    meta_total_tokens = n;

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

    /* Bigram — store in hash table style */
    typedef struct { int a, b; int count; } BC;
    BC *bcounts = (BC *)calloc(65536, sizeof(BC));
    int n_bc = 0;

    for (int i = 0; i + 1 < n; i++) {
        int a = tokens[i], b = tokens[i + 1];
        unsigned h = ((unsigned)a * 2654435761u ^ (unsigned)b) & 0xFFFF;
        for (int t = 0; t < 64; t++) {
            unsigned idx = (h + t) & 0xFFFF;
            if (bcounts[idx].count == 0) {
                bcounts[idx].a = a;
                bcounts[idx].b = b;
                bcounts[idx].count = 1;
                n_bc++;
                break;
            }
            if (bcounts[idx].a == a && bcounts[idx].b == b) {
                bcounts[idx].count++;
                break;
            }
        }
    }

    /* Convert to normalized bigrams */
    /* Group by 'a' and normalize */
    meta_n_bigrams = 0;
    for (int i = 0; i < 65536 && meta_n_bigrams < BIGRAM_CAP; i++) {
        if (bcounts[i].count > 0) {
            meta_bigrams[meta_n_bigrams].a = bcounts[i].a;
            meta_bigrams[meta_n_bigrams].b = bcounts[i].b;
            meta_bigrams[meta_n_bigrams].prob = (float)bcounts[i].count;
            meta_n_bigrams++;
        }
    }

    /* Normalize per 'a' */
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
    printf("  metaweights built: %d tokens, %d bigram entries\n", n, meta_n_bigrams);
}

static void meta_query_bigram(int prev, float *dist, int vs) {
    for (int i = 0; i < vs; i++) dist[i] = 1e-10f;
    for (int i = 0; i < meta_n_bigrams; i++) {
        if (meta_bigrams[i].a == prev && meta_bigrams[i].b < vs) {
            dist[meta_bigrams[i].b] = meta_bigrams[i].prob;
        }
    }
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

    float *probs = (float *)malloc(vocab_size * sizeof(float));
    float *bigram_dist = (float *)malloc(vocab_size * sizeof(float));

    for (int step = 0; step < max_tokens && gen_len < 4096; step++) {
        int last = generated[gen_len - 1];

        /* Query bigram metaweights */
        meta_query_bigram(last, bigram_dist, vocab_size);

        /* Build probability from metaweights */
        for (int i = 0; i < vocab_size; i++) {
            probs[i] = 2.0f * bigram_dist[i] + 0.01f * meta_unigram[i];
        }

        /* Temperature */
        for (int i = 0; i < vocab_size; i++)
            probs[i] /= temperature;

        softmax_inplace(probs, vocab_size);
        int chosen = sample_from_probs(probs, vocab_size);
        generated[gen_len++] = chosen;
    }

    free(probs);
    free(bigram_dist);

    bpe_decode(generated, gen_len, out, max_out);
}

static void generate_full(const int *prompt, int prompt_len, int max_tokens,
                           int vocab_size, float temperature, char *out, int max_out) {
    int generated[4096];
    int gen_len = prompt_len;
    memcpy(generated, prompt, prompt_len * sizeof(int));

    float *logits = (float *)malloc(vocab_size * sizeof(float));
    float *bigram_dist = (float *)malloc(vocab_size * sizeof(float));

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

        /* Dario field overlay */
        meta_query_bigram(last, bigram_dist, vocab_size);
        for (int i = 0; i < vocab_size; i++)
            logits[i] += 1.5f * bigram_dist[i];

        /* Temperature + sample */
        for (int i = 0; i < vocab_size; i++)
            logits[i] /= temperature;
        softmax_inplace(logits, vocab_size);

        int chosen = sample_from_probs(logits, vocab_size);
        generated[gen_len++] = chosen;
    }

    free(logits);
    free(bigram_dist);

    bpe_decode(generated, gen_len, out, max_out);
}

/* ───────────────────────── Main ───────────────────────── */

int main(void) {
    printf("============================================================\n");
    printf("  PostGPT (C) — metaweight BPE transformer\n");
    printf("  resonance is unbreakable\n");
    printf("============================================================\n");

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
    int n_tokens = bpe_learn(data, fsize, 1024, tokens);

    /* Build metaweights */
    printf("\n[3] Building metaweight probability space...\n");
    meta_build(tokens, n_tokens);

    /* Init transformer */
    printf("\n[4] Initializing PostGPT transformer...\n");
    weights_init(bpe_vocab_size);
    printf("  Initialized: vocab=%d, ctx=%d, embd=%d, heads=%d (content=%d, rrpram=%d), layers=%d\n",
           bpe_vocab_size, CONTEXT_LEN, N_EMBD, N_HEAD, N_CONTENT, N_RRPRAM, N_LAYER);

    /* Meta-generation */
    printf("\n[5] Meta-generation (metaweight mode):\n");
    printf("--------------------------------------------------\n");

    char output[4096];
    int seeds[][3] = {{tokens[0], tokens[1], tokens[2]},
                      {tokens[100], tokens[101], tokens[102]},
                      {tokens[500], tokens[501], tokens[502]}};

    for (int s = 0; s < 3; s++) {
        generate_meta(seeds[s], 3, 80, bpe_vocab_size, 0.75f, output, sizeof(output));
        printf("\n  sample %d: %.200s\n", s + 1, output);
    }

    /* Full generation with Dario field */
    printf("\n\n[6] Transformer + Dario field generation:\n");
    printf("--------------------------------------------------\n");
    generate_full(tokens, 4, 40, bpe_vocab_size, 0.8f, output, sizeof(output));
    printf("\n  output: %.300s\n", output);

    printf("\n============================================================\n");
    printf("  PostGPT complete. The metaweights remember.\n");
    printf("============================================================\n");

    free(data);
    free(tokens);
    return 0;
}
