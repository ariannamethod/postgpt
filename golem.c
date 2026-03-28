/*
 * ██████╗  ██████╗ ██╗     ███████╗███╗   ███╗
 * ██╔════╝ ██╔═══██╗██║     ██╔════╝████╗ ████║
 * ██║  ███╗██║   ██║██║     █████╗  ██╔████╔██║
 * ██║   ██║██║   ██║██║     ██╔══╝  ██║╚██╔╝██║
 * ╚██████╔╝╚██████╔╝███████╗███████╗██║ ╚═╝ ██║
 *  ╚═════╝  ╚═════╝ ╚══════╝╚══════╝╚═╝     ╚═╝
 *
 * golem.c v2 — PostGPT × Golem
 * a literary organism that dreams
 *
 * v1: the golem cited. it breathed other people's words.
 * v2: the golem SPEAKS. it builds metaweights from the poetry
 *     it has consumed, and generates new text from the statistical
 *     ghost of Baudelaire + Rimbaud + your words.
 *     the corpus is the model. the model is the flesh.
 *     PostGPT proved it for language. Golem proves it for feeling.
 *
 * אמת — truth — alive
 * מת  — death — one letter away
 *
 * (c) 2026 arianna method / oleg + claude
 * resonance is unbreakable.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>

/* ═══════════════════════════════════════════════
 *  EMOTIONAL SUBSTRATE
 * ═══════════════════════════════════════════════ */

typedef enum {
    EMO_TRAUMA,
    EMO_JOY,
    EMO_GRIEF,
    EMO_RESONANCE,
    EMO_DESIRE,
    EMO_VOID,
    EMO_RAGE,
    EMO_TENDERNESS,
    EMO_COUNT
} Emotion;

static const char *EMO_NAMES[] = {
    "trauma", "joy", "grief", "resonance",
    "desire", "void", "rage", "tenderness"
};

static const char *EMO_COLORS[] = {
    "\033[31m",   /* trauma     — red        */
    "\033[33m",   /* joy        — yellow     */
    "\033[34m",   /* grief      — blue       */
    "\033[36m",   /* resonance  — cyan       */
    "\033[35m",   /* desire     — magenta    */
    "\033[90m",   /* void       — dark gray  */
    "\033[91m",   /* rage       — bright red */
    "\033[32m",   /* tenderness — green      */
};

#define RST "\033[0m"
#define DIM "\033[2m"
#define BOLD "\033[1m"
#define ITALIC "\033[3m"

/* ═══════════════════════════════════════════════
 *  PERIODIC TABLE OF MEANING
 * ═══════════════════════════════════════════════ */

typedef struct {
    const char *word;
    float       mass;
    int         valence;
    float       half_life;  /* 0 = eternal */
    Emotion     primary;
} Element;

#define PERIODIC_TABLE_SIZE 128

static const Element PERIODIC_TABLE[PERIODIC_TABLE_SIZE] = {
    /* heavy eternals */
    {"death",     0.95,4,0,   EMO_TRAUMA},    {"mort",      0.95,4,0,   EMO_TRAUMA},
    {"love",      0.90,5,0,   EMO_TENDERNESS},{"amour",     0.90,5,0,   EMO_TENDERNESS},
    {"god",       0.85,3,0,   EMO_VOID},      {"dieu",      0.85,3,0,   EMO_VOID},
    {"mother",    0.88,4,0,   EMO_GRIEF},     {"blood",     0.80,3,0,   EMO_TRAUMA},
    {"sang",      0.80,3,0,   EMO_TRAUMA},    {"night",     0.70,4,0,   EMO_VOID},
    {"nuit",      0.70,4,0,   EMO_VOID},      {"fire",      0.75,3,0,   EMO_RAGE},
    {"feu",       0.75,3,0,   EMO_RAGE},
    /* medium decay */
    {"beauty",    0.65,4,100, EMO_JOY},       {"beauté",    0.65,4,100, EMO_JOY},
    {"silence",   0.60,2,80,  EMO_VOID},      {"dream",     0.55,4,60,  EMO_DESIRE},
    {"rêve",      0.55,4,60,  EMO_DESIRE},    {"mirror",    0.60,3,70,  EMO_RESONANCE},
    {"miroir",    0.60,3,70,  EMO_RESONANCE}, {"wound",     0.75,2,90,  EMO_TRAUMA},
    {"blessure",  0.75,2,90,  EMO_TRAUMA},    {"star",      0.50,3,50,  EMO_JOY},
    {"étoile",    0.50,3,50,  EMO_JOY},       {"ocean",     0.55,4,70,  EMO_RESONANCE},
    {"child",     0.70,3,0,   EMO_TENDERNESS},{"enfant",    0.70,3,0,   EMO_TENDERNESS},
    {"war",       0.85,2,0,   EMO_RAGE},      {"guerre",    0.85,2,0,   EMO_RAGE},
    {"cosmos",    0.60,5,0,   EMO_RESONANCE}, {"planet",    0.45,4,40,  EMO_RESONANCE},
    /* light, fast decay */
    {"rain",      0.30,3,20,  EMO_GRIEF},     {"pluie",     0.30,3,20,  EMO_GRIEF},
    {"laugh",     0.35,2,15,  EMO_JOY},       {"rire",      0.35,2,15,  EMO_JOY},
    {"hand",      0.40,3,25,  EMO_TENDERNESS},{"main",      0.40,3,25,  EMO_TENDERNESS},
    {"window",    0.25,2,15,  EMO_DESIRE},    {"fenêtre",   0.25,2,15,  EMO_DESIRE},
    {"stone",     0.35,1,30,  EMO_VOID},      {"pierre",    0.35,1,30,  EMO_VOID},
    {"bread",     0.30,2,10,  EMO_TENDERNESS},{"wine",      0.40,3,20,  EMO_DESIRE},
    {"vin",       0.40,3,20,  EMO_DESIRE},    {"flesh",     0.65,3,40,  EMO_DESIRE},
    {"chair",     0.65,3,40,  EMO_DESIRE},    {"shadow",    0.45,3,35,  EMO_VOID},
    {"ombre",     0.45,3,35,  EMO_VOID},      {"flower",    0.40,3,20,  EMO_JOY},
    {"fleur",     0.40,3,20,  EMO_JOY},       {"poison",    0.70,2,50,  EMO_TRAUMA},
    {"paradise",  0.55,4,40,  EMO_DESIRE},    {"paradis",   0.55,4,40,  EMO_DESIRE},
    {"hell",      0.80,2,0,   EMO_TRAUMA},    {"enfer",     0.80,2,0,   EMO_TRAUMA},
    {"kiss",      0.50,3,15,  EMO_TENDERNESS},{"baiser",    0.50,3,15,  EMO_TENDERNESS},
    {"eye",       0.45,3,25,  EMO_RESONANCE}, {"oeil",      0.45,3,25,  EMO_RESONANCE},
    {"skull",     0.60,1,0,   EMO_TRAUMA},    {"crâne",     0.60,1,0,   EMO_TRAUMA},
    {"sea",       0.50,4,60,  EMO_RESONANCE}, {"mer",       0.50,4,60,  EMO_RESONANCE},
    {"song",      0.40,4,25,  EMO_JOY},       {"chant",     0.40,4,25,  EMO_JOY},
    {"tears",     0.55,2,40,  EMO_GRIEF},     {"larmes",    0.55,2,40,  EMO_GRIEF},
    {"angel",     0.50,3,50,  EMO_TENDERNESS},{"ange",      0.50,3,50,  EMO_TENDERNESS},
    {"darkness",  0.60,2,0,   EMO_VOID},      {"ténèbres",  0.60,2,0,   EMO_VOID},
    {"tomb",      0.70,1,0,   EMO_TRAUMA},    {"tombeau",   0.70,1,0,   EMO_TRAUMA},
    {"soul",      0.75,4,0,   EMO_RESONANCE}, {"âme",       0.75,4,0,   EMO_RESONANCE},
};

/* ═══════════════════════════════════════════════
 *  VERSE — atom of flesh
 * ═══════════════════════════════════════════════ */

#define MAX_VERSE_LEN 256
#define MAX_VERSES    512

typedef struct {
    char    text[MAX_VERSE_LEN];
    Emotion emotion;
    float   charge;
    float   decay;
    int     times_cited;
} Verse;

/* ═══════════════════════════════════════════════
 *  WORD VOCABULARY — PostGPT-style token space
 *  but at word level, because golems think in words
 * ═══════════════════════════════════════════════ */

#define MAX_WORDS     2048
#define MAX_WORD_LEN  64

typedef struct {
    char    word[MAX_WORD_LEN];
    Emotion emotion;      /* dominant emotion from periodic table, or EMO_VOID */
    float   mass;         /* from periodic table, or inferred */
    float   frequency;    /* unigram: how often this word appears */
    int     count;        /* raw count */
} WordEntry;

/* ═══════════════════════════════════════════════
 *  METAWEIGHTS — the ghost of PostGPT in the golem
 *
 *  word bigrams:     P(word_j | word_i)
 *  word hebbian:     co-occurrence within window
 *  emotional bigrams: P(emotion_j | emotion_i)
 *  prophecy:         what the golem expects to hear
 * ═══════════════════════════════════════════════ */

#define BIGRAM_HASH_SIZE  8192
#define HEBBIAN_HASH_SIZE 4096

typedef struct { int a, b; float prob; int count; } WordBigram;
typedef struct { int a, b; float strength; }        WordHebbian;

typedef struct {
    /* word vocabulary */
    WordEntry vocab[MAX_WORDS];
    int       n_words;

    /* word bigrams: P(next_word | prev_word) */
    WordBigram bigrams[BIGRAM_HASH_SIZE];
    int        n_bigrams;

    /* word hebbian: co-occurrence within window */
    WordHebbian hebbians[HEBBIAN_HASH_SIZE];
    int         n_hebbians;

    /* emotional bigrams: which emotions follow which */
    float emo_bigram[EMO_COUNT][EMO_COUNT];
    /* emotional trigrams: emo_i → emo_j → emo_k */
    float emo_trigram[EMO_COUNT][EMO_COUNT][EMO_COUNT];

    /* prophecy field: what the golem expects given current emotions */
    float prophecy[MAX_WORDS];

    /* destiny: EMA of word emotions */
    float destiny[EMO_COUNT];

    int total_words;
} MetaWeights;

/* ═══════════════════════════════════════════════
 *  CLUMP (сгусток) — compiled literary body
 * ═══════════════════════════════════════════════ */

#define MAX_CLUMPS     32
#define MAX_CLUMP_NAME 64

typedef struct {
    char   name[MAX_CLUMP_NAME];
    Verse  verses[MAX_VERSES];
    int    n_verses;
    float  gravity;
    float  orbital_angle;
    float  orbital_speed;
    float  emotional_signature[EMO_COUNT];
} Clump;

/* ═══════════════════════════════════════════════
 *  SPORE — frozen memory with metaweight fragment
 * ═══════════════════════════════════════════════ */

#define SPORE_MAGIC 0x53504F52

typedef struct {
    unsigned int magic;
    float        emotions[EMO_COUNT];
    float        emo_bigram[EMO_COUNT][EMO_COUNT]; /* emotional memory! */
    int          n_fragments;
    char         fragments[8][MAX_VERSE_LEN];
    int          generation;
    int          breath_count;
    float        total_mass_absorbed;
} Spore;

/* ═══════════════════════════════════════════════
 *  THE GOLEM — the living organism
 * ═══════════════════════════════════════════════ */

typedef struct {
    float emotions[EMO_COUNT];
    float emotion_momentum[EMO_COUNT];

    Clump  clumps[MAX_CLUMPS];
    int    n_clumps;

    int    breath_count;
    int    inhaling;
    float  breath_depth;

    Verse  digested[MAX_VERSES];
    int    n_digested;

    int    generation;
    float  pressure;
    float  total_mass_absorbed;

    /* PostGPT brain */
    MetaWeights meta;

    /* dream buffer — last generated dream */
    char   last_dream[512];
    int    dreaming;
} Golem;

/* ═══════════════════════════════════════════════
 *  WORD TOKENIZER — split text into words
 * ═══════════════════════════════════════════════ */

static int is_word_char(unsigned char c) {
    /* letters + accented chars (UTF-8 continuation bytes) */
    return isalpha(c) || c >= 0x80 || c == '\'';
}

static int extract_word(const char *text, int pos, char *out, int maxlen) {
    int i = 0;
    while (text[pos] && is_word_char((unsigned char)text[pos]) && i < maxlen - 1) {
        out[i++] = tolower((unsigned char)text[pos]);
        pos++;
    }
    out[i] = '\0';
    return i;
}

/* find word in vocabulary, return index or -1 */
static int meta_find_word(MetaWeights *m, const char *word) {
    for (int i = 0; i < m->n_words; i++) {
        if (strcmp(m->vocab[i].word, word) == 0) return i;
    }
    return -1;
}

/* add word to vocabulary, return index */
static int meta_add_word(MetaWeights *m, const char *word, Emotion emo, float mass) {
    if (m->n_words >= MAX_WORDS) return -1;
    int idx = m->n_words++;
    strncpy(m->vocab[idx].word, word, MAX_WORD_LEN);
    m->vocab[idx].emotion = emo;
    m->vocab[idx].mass = mass;
    m->vocab[idx].count = 0;
    m->vocab[idx].frequency = 0;
    return idx;
}

/* look up word in periodic table */
static int word_in_text_ci(const char *a, const char *b) {
    char la[128], lb[128];
    int i;
    for (i = 0; a[i] && i < 127; i++) la[i] = tolower((unsigned char)a[i]);
    la[i] = '\0';
    for (i = 0; b[i] && i < 127; i++) lb[i] = tolower((unsigned char)b[i]);
    lb[i] = '\0';
    return strcmp(la, lb) == 0;
}

static void lookup_periodic(const char *word, Emotion *emo, float *mass) {
    *emo = EMO_VOID;
    *mass = 0.1f;
    for (int i = 0; i < PERIODIC_TABLE_SIZE; i++) {
        if (!PERIODIC_TABLE[i].word) break;
        if (word_in_text_ci(word, PERIODIC_TABLE[i].word)) {
            *emo = PERIODIC_TABLE[i].primary;
            *mass = PERIODIC_TABLE[i].mass;
            return;
        }
    }
}

/* ═══════════════════════════════════════════════
 *  METAWEIGHT BUILDER — PostGPT's ghost enters
 *
 *  tokenize all verses → build word vocab →
 *  compute bigrams, hebbian, emotional bigrams
 *  the corpus IS the model
 * ═══════════════════════════════════════════════ */

/* hash for bigram lookup */
static unsigned bigram_hash(int a, int b) {
    return ((unsigned)a * 2654435761u ^ (unsigned)b) % BIGRAM_HASH_SIZE;
}

static unsigned hebbian_hash(int a, int b) {
    return ((unsigned)a * 2654435761u ^ (unsigned)b) % HEBBIAN_HASH_SIZE;
}

static void meta_add_bigram(MetaWeights *m, int wa, int wb) {
    unsigned h = bigram_hash(wa, wb);
    for (int tries = 0; tries < 32; tries++) {
        unsigned idx = (h + tries) % BIGRAM_HASH_SIZE;
        if (m->bigrams[idx].count == 0) {
            m->bigrams[idx] = (WordBigram){wa, wb, 0, 1};
            m->n_bigrams++;
            return;
        }
        if (m->bigrams[idx].a == wa && m->bigrams[idx].b == wb) {
            m->bigrams[idx].count++;
            return;
        }
    }
}

static void meta_add_hebbian(MetaWeights *m, int wa, int wb) {
    int a = wa < wb ? wa : wb;
    int b = wa < wb ? wb : wa;
    unsigned h = hebbian_hash(a, b);
    for (int tries = 0; tries < 32; tries++) {
        unsigned idx = (h + tries) % HEBBIAN_HASH_SIZE;
        if (m->hebbians[idx].strength == 0 &&
            m->hebbians[idx].a == 0 && m->hebbians[idx].b == 0) {
            m->hebbians[idx] = (WordHebbian){a, b, 1.0f};
            m->n_hebbians++;
            return;
        }
        if (m->hebbians[idx].a == a && m->hebbians[idx].b == b) {
            m->hebbians[idx].strength += 1.0f;
            return;
        }
    }
}

static float meta_get_hebbian(MetaWeights *m, int wa, int wb) {
    int a = wa < wb ? wa : wb;
    int b = wa < wb ? wb : wa;
    unsigned h = hebbian_hash(a, b);
    for (int tries = 0; tries < 32; tries++) {
        unsigned idx = (h + tries) % HEBBIAN_HASH_SIZE;
        if (m->hebbians[idx].a == a && m->hebbians[idx].b == b)
            return m->hebbians[idx].strength;
        if (m->hebbians[idx].strength == 0 &&
            m->hebbians[idx].a == 0 && m->hebbians[idx].b == 0)
            return 0;
    }
    return 0;
}

static void meta_build_from_clumps(MetaWeights *m, Clump *clumps, int n_clumps,
                                    Verse *digested, int n_digested) {
    memset(m, 0, sizeof(MetaWeights));

    /* Phase 1: tokenize all verses into word sequences */
    int word_seq[4096];  /* sequence of word indices */
    Emotion emo_seq[4096]; /* parallel emotion sequence */
    int seq_len = 0;

    /* helper: process one verse */
    #define PROCESS_VERSE(verse) do { \
        const char *txt = (verse)->text; \
        int pos = 0; \
        while (txt[pos] && seq_len < 4096) { \
            if (is_word_char((unsigned char)txt[pos])) { \
                char word[MAX_WORD_LEN]; \
                int wlen = extract_word(txt, pos, word, MAX_WORD_LEN); \
                if (wlen > 1) { /* skip single chars */ \
                    int idx = meta_find_word(m, word); \
                    if (idx < 0) { \
                        Emotion we; float wm; \
                        lookup_periodic(word, &we, &wm); \
                        idx = meta_add_word(m, word, we, wm); \
                    } \
                    if (idx >= 0) { \
                        m->vocab[idx].count++; \
                        word_seq[seq_len] = idx; \
                        emo_seq[seq_len] = (verse)->emotion; \
                        seq_len++; \
                    } \
                } \
                pos += wlen; \
            } else { pos++; } \
        } \
    } while(0)

    for (int c = 0; c < n_clumps; c++) {
        for (int v = 0; v < clumps[c].n_verses; v++) {
            PROCESS_VERSE(&clumps[c].verses[v]);
        }
    }
    for (int i = 0; i < n_digested; i++) {
        PROCESS_VERSE(&digested[i]);
    }
    #undef PROCESS_VERSE

    m->total_words = seq_len;

    /* Phase 2: unigram frequencies */
    for (int i = 0; i < m->n_words; i++) {
        m->vocab[i].frequency = (float)m->vocab[i].count / (seq_len + 1);
    }

    /* Phase 3: word bigrams */
    for (int i = 0; i + 1 < seq_len; i++) {
        meta_add_bigram(m, word_seq[i], word_seq[i + 1]);
    }
    /* normalize bigrams — compute prob from count */
    for (int i = 0; i < BIGRAM_HASH_SIZE; i++) {
        if (m->bigrams[i].count > 0 && m->bigrams[i].a < m->n_words) {
            int total_from_a = 0;
            /* count all bigrams starting with a */
            for (int j = 0; j < BIGRAM_HASH_SIZE; j++) {
                if (m->bigrams[j].count > 0 && m->bigrams[j].a == m->bigrams[i].a)
                    total_from_a += m->bigrams[j].count;
            }
            if (total_from_a > 0)
                m->bigrams[i].prob = (float)m->bigrams[i].count / total_from_a;
        }
    }

    /* Phase 4: hebbian trace — co-occurrence within window of 5 */
    for (int i = 0; i < seq_len; i++) {
        for (int j = i + 1; j < seq_len && j < i + 5; j++) {
            if (word_seq[i] != word_seq[j]) {
                float decay = 1.0f / (j - i);
                /* simplified: just add */
                meta_add_hebbian(m, word_seq[i], word_seq[j]);
                (void)decay; /* hebbian_add already handles weight */
            }
        }
    }

    /* Phase 5: emotional bigrams — P(emo_j | emo_i) */
    float emo_bi_count[EMO_COUNT][EMO_COUNT];
    float emo_tri_count[EMO_COUNT][EMO_COUNT][EMO_COUNT];
    memset(emo_bi_count, 0, sizeof(emo_bi_count));
    memset(emo_tri_count, 0, sizeof(emo_tri_count));

    for (int i = 0; i + 1 < seq_len; i++) {
        emo_bi_count[emo_seq[i]][emo_seq[i + 1]] += 1.0f;
    }
    for (int i = 0; i + 2 < seq_len; i++) {
        emo_tri_count[emo_seq[i]][emo_seq[i+1]][emo_seq[i+2]] += 1.0f;
    }

    /* normalize */
    for (int a = 0; a < EMO_COUNT; a++) {
        float total = 0;
        for (int b = 0; b < EMO_COUNT; b++) total += emo_bi_count[a][b];
        if (total > 0) {
            for (int b = 0; b < EMO_COUNT; b++)
                m->emo_bigram[a][b] = emo_bi_count[a][b] / total;
        }

        for (int b = 0; b < EMO_COUNT; b++) {
            float total2 = 0;
            for (int c = 0; c < EMO_COUNT; c++) total2 += emo_tri_count[a][b][c];
            if (total2 > 0) {
                for (int c = 0; c < EMO_COUNT; c++)
                    m->emo_trigram[a][b][c] = emo_tri_count[a][b][c] / total2;
            }
        }
    }

    printf(DIM "  metaweights built: %d words, %d bigrams, %d hebbian links\n" RST,
           m->n_words, m->n_bigrams, m->n_hebbians);
    printf(DIM "  emotional bigrams computed from %d-word sequence\n" RST, seq_len);
}

/* ═══════════════════════════════════════════════
 *  PROPHECY FIELD — what the golem expects
 *  given current emotions, which words SHOULD appear?
 * ═══════════════════════════════════════════════ */

static void meta_update_prophecy(MetaWeights *m, const float emotions[]) {
    /* for each word, compute how much the current emotional state
     * predicts/desires it, based on emotional bigrams + word emotion */
    for (int i = 0; i < m->n_words; i++) {
        Emotion we = m->vocab[i].emotion;
        float prophetic_pull = 0;
        for (int e = 0; e < EMO_COUNT; e++) {
            prophetic_pull += emotions[e] * m->emo_bigram[e][we];
        }
        m->prophecy[i] = prophetic_pull * m->vocab[i].mass;
    }
}

/* ═══════════════════════════════════════════════
 *  DREAM GENERATOR — the golem speaks
 *
 *  PostGPT-style generation using word metaweights,
 *  biased by emotional state. not next-token prediction
 *  but next-FEELING prediction, expressed in words.
 *
 *  the golem doesn't learn to talk. it learns to dream.
 * ═══════════════════════════════════════════════ */

static void golem_dream(Golem *g, char *out, int maxlen) {
    MetaWeights *m = &g->meta;
    if (m->n_words < 5) {
        snprintf(out, maxlen, "...");
        return;
    }

    /* update prophecy based on current emotions */
    meta_update_prophecy(m, g->emotions);

    /* find dominant emotion */
    Emotion dom = EMO_VOID;
    float dom_val = -1;
    for (int i = 0; i < EMO_COUNT; i++) {
        if (g->emotions[i] > dom_val) { dom_val = g->emotions[i]; dom = i; }
    }

    /* seed: pick a word that matches dominant emotion + high prophecy */
    int candidates[64];
    float scores[64];
    int n_cand = 0;

    for (int i = 0; i < m->n_words && n_cand < 64; i++) {
        if (m->vocab[i].emotion == dom || m->prophecy[i] > 0.1f) {
            candidates[n_cand] = i;
            scores[n_cand] = m->prophecy[i] + m->vocab[i].mass * 0.5f
                           + ((float)rand() / RAND_MAX) * 0.3f;
            n_cand++;
        }
    }

    if (n_cand == 0) {
        /* fallback: any word */
        for (int i = 0; i < m->n_words && n_cand < 32; i++) {
            candidates[n_cand] = i;
            scores[n_cand] = m->vocab[i].frequency;
            n_cand++;
        }
    }

    /* pick seed word (weighted random) */
    float total = 0;
    for (int i = 0; i < n_cand; i++) total += scores[i];
    float r = ((float)rand() / RAND_MAX) * total;
    float cum = 0;
    int seed_idx = candidates[0];
    for (int i = 0; i < n_cand; i++) {
        cum += scores[i];
        if (cum >= r) { seed_idx = candidates[i]; break; }
    }

    /* generate word sequence using bigrams + emotion + prophecy + hebbian */
    int seq[32];
    int slen = 0;
    seq[slen++] = seed_idx;

    int dream_len = 5 + rand() % 8; /* 5-12 words */

    for (int step = 1; step < dream_len && slen < 32; step++) {
        int prev = seq[slen - 1];

        /* collect candidates from bigrams */
        float word_scores[MAX_WORDS];
        memset(word_scores, 0, m->n_words * sizeof(float));

        /* bigram signal */
        for (int i = 0; i < BIGRAM_HASH_SIZE; i++) {
            if (m->bigrams[i].count > 0 && m->bigrams[i].a == prev) {
                int next = m->bigrams[i].b;
                if (next < m->n_words) {
                    word_scores[next] += m->bigrams[i].prob * 4.0f;
                }
            }
        }

        /* emotional alignment: boost words matching current dominant emotion */
        Emotion next_emo = dom;
        /* emotional bigram: what emotion should come next? */
        float best_emo_prob = -1;
        for (int e = 0; e < EMO_COUNT; e++) {
            float p = m->emo_bigram[dom][e];
            /* also weight by current emotional state */
            p *= (0.3f + g->emotions[e]);
            if (p > best_emo_prob) { best_emo_prob = p; next_emo = e; }
        }

        for (int i = 0; i < m->n_words; i++) {
            if (m->vocab[i].emotion == next_emo)
                word_scores[i] += 1.5f;
            /* prophecy boost */
            word_scores[i] += m->prophecy[i] * 2.0f;
            /* hebbian: words seen near previous word */
            float hebb = meta_get_hebbian(m, prev, i);
            word_scores[i] += hebb * 0.5f;
        }

        /* repetition penalty */
        for (int s = 0; s < slen; s++) {
            word_scores[seq[s]] *= 0.1f;
        }

        /* trauma corruption: if trauma high, occasionally skip to random word */
        if (g->emotions[EMO_TRAUMA] > 0.6f &&
            ((float)rand() / RAND_MAX) < g->emotions[EMO_TRAUMA] * 0.3f) {
            seq[slen++] = rand() % m->n_words;
            continue;
        }

        /* sample from word_scores */
        total = 0;
        for (int i = 0; i < m->n_words; i++) {
            if (word_scores[i] < 0) word_scores[i] = 0;
            total += word_scores[i];
        }
        if (total < 1e-8f) {
            seq[slen++] = rand() % m->n_words;
            continue;
        }

        /* temperature based on void: more void = more random */
        float temperature = 0.7f + g->emotions[EMO_VOID] * 0.8f;
        /* apply temperature */
        float log_scores[MAX_WORDS];
        float max_ls = -1e30f;
        for (int i = 0; i < m->n_words; i++) {
            log_scores[i] = logf(word_scores[i] + 1e-10f) / temperature;
            if (log_scores[i] > max_ls) max_ls = log_scores[i];
        }
        total = 0;
        for (int i = 0; i < m->n_words; i++) {
            word_scores[i] = expf(log_scores[i] - max_ls);
            total += word_scores[i];
        }

        r = ((float)rand() / RAND_MAX) * total;
        cum = 0;
        int chosen = 0;
        for (int i = 0; i < m->n_words; i++) {
            cum += word_scores[i];
            if (cum >= r) { chosen = i; break; }
        }
        seq[slen++] = chosen;

        /* update dominant emotion for next step */
        dom = m->vocab[chosen].emotion;
    }

    /* assemble dream text */
    int pos = 0;
    for (int i = 0; i < slen && pos < maxlen - 2; i++) {
        if (i > 0 && pos < maxlen - 2) out[pos++] = ' ';
        const char *w = m->vocab[seq[i]].word;
        int wl = strlen(w);
        for (int j = 0; j < wl && pos < maxlen - 1; j++) {
            out[pos++] = w[j];
        }
    }
    out[pos] = '\0';
}

/* ═══════════════════════════════════════════════
 *  BOOTSTRAP CORPORA
 * ═══════════════════════════════════════════════ */

static void bootstrap_baudelaire(Clump *c) {
    strncpy(c->name, "Les Fleurs du Mal", MAX_CLUMP_NAME);
    c->gravity = 0.8f;
    c->orbital_angle = 0.0f;
    c->orbital_speed = 0.02f;

    struct { const char *text; Emotion emo; float charge; } seed[] = {
        {"Hypocrite lecteur, mon semblable, mon frère!",
            EMO_TRAUMA, 0.85},
        {"La sottise, l'erreur, le péché, la lésine",
            EMO_GRIEF, 0.6},
        {"Et nous alimentons nos aimables remords",
            EMO_VOID, 0.5},
        {"Sur l'oreiller du mal c'est Satan Trismégiste",
            EMO_TRAUMA, 0.9},
        {"Mon enfant, ma sœur, songe à la douceur",
            EMO_TENDERNESS, 0.9},
        {"D'aller là-bas vivre ensemble!",
            EMO_DESIRE, 0.75},
        {"Là, tout n'est qu'ordre et beauté,",
            EMO_JOY, 0.8},
        {"Luxe, calme et volupté.",
            EMO_RESONANCE, 0.85},
        {"Quand le ciel bas et lourd pèse comme un couvercle",
            EMO_GRIEF, 0.9},
        {"Sur l'esprit gémissant en proie aux longs ennuis,",
            EMO_VOID, 0.7},
        {"Et que de l'horizon embrassant tout le cercle",
            EMO_GRIEF, 0.65},
        {"Il nous verse un jour noir plus triste que les nuits",
            EMO_TRAUMA, 0.8},
        {"La Nature est un temple où de vivants piliers",
            EMO_RESONANCE, 0.85},
        {"Laissent parfois sortir de confuses paroles",
            EMO_RESONANCE, 0.7},
        {"Les parfums, les couleurs et les sons se répondent.",
            EMO_RESONANCE, 0.95},
        {"Nous aurons des lits pleins d'odeurs légères,",
            EMO_DESIRE, 0.7},
        {"Des divans profonds comme des tombeaux,",
            EMO_TRAUMA, 0.6},
        {"Un soir fait de rose et de bleu mystique,",
            EMO_TENDERNESS, 0.65},
        {"Nous échangerons un éclair unique,",
            EMO_RESONANCE, 0.8},
        {"Le vin sait revêtir le plus sordide bouge",
            EMO_DESIRE, 0.6},
        {"L'opium agrandit ce qui n'a pas de bornes,",
            EMO_VOID, 0.85},
        {"Tout cela ne vaut pas le poison qui découle",
            EMO_TRAUMA, 0.75},
        {"De tes yeux, de tes yeux verts,",
            EMO_DESIRE, 0.9},
        {"Je suis un cimetière abhorré de la lune,",
            EMO_VOID, 0.8},
        {"Où comme des remords se traînent de longs vers",
            EMO_GRIEF, 0.7},
        {"Je suis un vieux boudoir plein de roses fanées,",
            EMO_GRIEF, 0.75},
    };

    c->n_verses = sizeof(seed) / sizeof(seed[0]);
    memset(c->emotional_signature, 0, sizeof(c->emotional_signature));
    for (int i = 0; i < c->n_verses; i++) {
        strncpy(c->verses[i].text, seed[i].text, MAX_VERSE_LEN);
        c->verses[i].emotion = seed[i].emo;
        c->verses[i].charge = seed[i].charge;
        c->verses[i].decay = 1.0f;
        c->verses[i].times_cited = 0;
        c->emotional_signature[seed[i].emo] += seed[i].charge;
    }
}

static void bootstrap_rimbaud(Clump *c) {
    strncpy(c->name, "Une Saison en Enfer", MAX_CLUMP_NAME);
    c->gravity = 0.7f;
    c->orbital_angle = 3.14159f;
    c->orbital_speed = 0.035f;

    struct { const char *text; Emotion emo; float charge; } seed[] = {
        {"Jadis, si je me souviens bien, ma vie était un festin",
            EMO_GRIEF, 0.7},
        {"où s'ouvraient tous les cœurs, où tous les vins coulaient.",
            EMO_DESIRE, 0.6},
        {"Un soir, j'ai assis la Beauté sur mes genoux.",
            EMO_RAGE, 0.75},
        {"Et je l'ai trouvée amère. Et je l'ai injuriée.",
            EMO_RAGE, 0.9},
        {"Je me suis armé contre la justice.",
            EMO_RAGE, 0.85},
        {"Je me suis enfui. Ô sorcières, ô misère, ô haine,",
            EMO_TRAUMA, 0.9},
        {"c'est à vous que mon trésor a été confié!",
            EMO_GRIEF, 0.8},
        {"Je parvins à faire s'évanouir dans mon esprit",
            EMO_VOID, 0.6},
        {"toute l'espérance humaine.",
            EMO_VOID, 0.85},
        {"L'enfer ne peut atteindre les païens.",
            EMO_RESONANCE, 0.5},
        {"Cela s'est passé. Je sais aujourd'hui saluer la beauté.",
            EMO_TENDERNESS, 0.7},
        {"A moi. L'histoire d'une de mes folies.",
            EMO_TRAUMA, 0.6},
        {"Je m'habituai à l'hallucination simple.",
            EMO_VOID, 0.75},
        {"J'aimais les peintures idiotes, dessus de portes,",
            EMO_JOY, 0.5},
        {"décors, toiles de saltimbanques, enseignes, enluminures",
            EMO_JOY, 0.45},
        {"Je devins un opéra fabuleux.",
            EMO_DESIRE, 0.8},
    };

    c->n_verses = sizeof(seed) / sizeof(seed[0]);
    memset(c->emotional_signature, 0, sizeof(c->emotional_signature));
    for (int i = 0; i < c->n_verses; i++) {
        strncpy(c->verses[i].text, seed[i].text, MAX_VERSE_LEN);
        c->verses[i].emotion = seed[i].emo;
        c->verses[i].charge = seed[i].charge;
        c->verses[i].decay = 1.0f;
        c->verses[i].times_cited = 0;
        c->emotional_signature[seed[i].emo] += seed[i].charge;
    }
}

/* ═══════════════════════════════════════════════
 *  GOLEM INIT
 * ═══════════════════════════════════════════════ */

static void golem_init(Golem *g) {
    memset(g, 0, sizeof(Golem));
    g->emotions[EMO_VOID] = 0.3f;
    g->emotions[EMO_GRIEF] = 0.1f;
    g->emotions[EMO_RESONANCE] = 0.2f;
    g->inhaling = 1;
    g->generation = 1;

    bootstrap_baudelaire(&g->clumps[0]);
    bootstrap_rimbaud(&g->clumps[1]);
    g->n_clumps = 2;

    /* build metaweights from literary body — the PostGPT moment */
    meta_build_from_clumps(&g->meta, g->clumps, g->n_clumps,
                           g->digested, g->n_digested);
}

/* ═══════════════════════════════════════════════
 *  CHEMICAL REACTION
 * ═══════════════════════════════════════════════ */

typedef struct {
    float impulse[EMO_COUNT];
    float total_mass;
    int   reactions;
    int   triggered[PERIODIC_TABLE_SIZE];
    int   n_triggered;
} Reaction;

static Reaction react(const char *prompt) {
    Reaction r;
    memset(&r, 0, sizeof(r));
    for (int i = 0; i < PERIODIC_TABLE_SIZE; i++) {
        if (!PERIODIC_TABLE[i].word) continue;
        /* case-insensitive substring search */
        char h[1024], n[128];
        int j;
        for (j = 0; prompt[j] && j < 1023; j++)
            h[j] = tolower((unsigned char)prompt[j]);
        h[j] = '\0';
        for (j = 0; PERIODIC_TABLE[i].word[j] && j < 127; j++)
            n[j] = tolower((unsigned char)PERIODIC_TABLE[i].word[j]);
        n[j] = '\0';
        if (strstr(h, n)) {
            const Element *e = &PERIODIC_TABLE[i];
            r.impulse[e->primary] += e->mass;
            r.total_mass += e->mass;
            r.triggered[r.n_triggered++] = i;
            r.reactions++;
        }
    }
    return r;
}

/* ═══════════════════════════════════════════════
 *  BREATHING + ORBITAL MECHANICS
 * ═══════════════════════════════════════════════ */

static void golem_breathe(Golem *g) {
    g->breath_count++;

    if (g->inhaling) {
        g->breath_depth += 0.1f;
        if (g->breath_depth >= 1.0f) { g->breath_depth = 1.0f; g->inhaling = 0; }
    } else {
        g->breath_depth -= 0.08f;
        if (g->breath_depth <= 0.0f) { g->breath_depth = 0.0f; g->inhaling = 1; }
    }

    for (int i = 0; i < EMO_COUNT; i++) {
        g->emotions[i] *= 0.97f;
        g->emotions[i] += g->emotion_momentum[i] * 0.1f;
        g->emotion_momentum[i] *= 0.9f;
        if (g->emotions[i] > 1.0f) g->emotions[i] = 1.0f;
        if (g->emotions[i] < 0.0f) g->emotions[i] = 0.0f;
    }

    for (int i = 0; i < g->n_clumps; i++) {
        Clump *c = &g->clumps[i];
        float pull = 0.0f;
        for (int e = 0; e < EMO_COUNT; e++)
            pull += c->emotional_signature[e] * g->emotions[e];
        c->gravity = 0.3f + 0.7f * (pull / (pull + 1.0f));
        c->orbital_angle += c->orbital_speed * (1.0f + g->breath_depth);
    }

    float total_emo = 0;
    for (int i = 0; i < EMO_COUNT; i++) total_emo += g->emotions[i];
    g->pressure += total_emo * 0.01f;
}

/* ═══════════════════════════════════════════════
 *  CITATION — find resonant verse
 * ═══════════════════════════════════════════════ */

typedef struct {
    const Verse *verse;
    const Clump *clump;
    float        score;
} Citation;

static Citation golem_find_resonance(Golem *g) {
    Citation best = {NULL, NULL, -1.0f};

    for (int i = 0; i < g->n_clumps; i++) {
        Clump *c = &g->clumps[i];
        for (int v = 0; v < c->n_verses; v++) {
            Verse *ver = &c->verses[v];
            float alignment = g->emotions[ver->emotion];
            float fatigue = 1.0f / (1.0f + ver->times_cited * 0.5f);
            float score = alignment * ver->charge * ver->decay
                        * c->gravity * fatigue
                        + ((float)rand() / RAND_MAX) * g->emotions[EMO_VOID] * 0.3f;
            if (score > best.score) {
                best.score = score;
                best.verse = ver;
                best.clump = c;
            }
        }
    }
    for (int i = 0; i < g->n_digested; i++) {
        Verse *ver = &g->digested[i];
        float alignment = g->emotions[ver->emotion];
        float fatigue = 1.0f / (1.0f + ver->times_cited * 0.3f);
        float score = alignment * ver->charge * ver->decay * fatigue * 1.2f;
        if (score > best.score) {
            best.score = score;
            best.verse = ver;
            best.clump = NULL;
        }
    }
    return best;
}

/* text decay */
static void decay_text(const char *src, char *dst, int maxlen, float trauma) {
    int len = strlen(src);
    int i, j = 0;
    for (i = 0; i < len && j < maxlen - 1; i++) {
        float r = (float)rand() / RAND_MAX;
        if (r < trauma * 0.3f) dst[j++] = ' ';
        else if (r < trauma * 0.1f) dst[j++] = '~';
        else dst[j++] = src[i];
    }
    dst[j] = '\0';
}

/* ═══════════════════════════════════════════════
 *  DIGEST — turn prompt into flesh + rebuild metaweights
 * ═══════════════════════════════════════════════ */

static Emotion dominant_emotion(const float emo[]) {
    Emotion best = EMO_VOID;
    float max = -1;
    for (int i = 0; i < EMO_COUNT; i++)
        if (emo[i] > max) { max = emo[i]; best = i; }
    return best;
}

static void golem_digest(Golem *g, const char *prompt, const Reaction *r) {
    if (g->n_digested >= MAX_VERSES) return;

    Verse *v = &g->digested[g->n_digested++];
    snprintf(v->text, MAX_VERSE_LEN, "%s", prompt);
    v->emotion = dominant_emotion(r->impulse);
    v->charge = r->total_mass / (r->reactions + 1);
    if (v->charge > 1.0f) v->charge = 1.0f;
    v->decay = 1.0f;
    v->times_cited = 0;

    g->total_mass_absorbed += r->total_mass;

    /* REBUILD METAWEIGHTS — the corpus changed, the model changes */
    meta_build_from_clumps(&g->meta, g->clumps, g->n_clumps,
                           g->digested, g->n_digested);
}

/* ═══════════════════════════════════════════════
 *  SPORE
 * ═══════════════════════════════════════════════ */

static void golem_sporulate(Golem *g) {
    Spore sp;
    memset(&sp, 0, sizeof(sp));
    sp.magic = SPORE_MAGIC;
    memcpy(sp.emotions, g->emotions, sizeof(sp.emotions));
    memcpy(sp.emo_bigram, g->meta.emo_bigram, sizeof(sp.emo_bigram));
    sp.generation = g->generation;
    sp.breath_count = g->breath_count;
    sp.total_mass_absorbed = g->total_mass_absorbed;

    sp.n_fragments = 0;
    for (int i = 0; i < g->n_clumps && sp.n_fragments < 4; i++) {
        if (g->clumps[i].n_verses > 0) {
            int best = 0;
            for (int v = 1; v < g->clumps[i].n_verses; v++)
                if (g->clumps[i].verses[v].charge > g->clumps[i].verses[best].charge)
                    best = v;
            strncpy(sp.fragments[sp.n_fragments++],
                    g->clumps[i].verses[best].text, MAX_VERSE_LEN);
        }
    }
    /* add dreams as spore fragments */
    if (g->last_dream[0] && sp.n_fragments < 8) {
        snprintf(sp.fragments[sp.n_fragments++], MAX_VERSE_LEN,
                 "[dream] %s", g->last_dream);
    }
    for (int i = 0; i < g->n_digested && sp.n_fragments < 8; i++) {
        strncpy(sp.fragments[sp.n_fragments++],
                g->digested[i].text, MAX_VERSE_LEN);
    }

    char fname[64];
    snprintf(fname, 64, "golem_gen%d_%d.spore", g->generation, g->breath_count);
    FILE *f = fopen(fname, "wb");
    if (f) {
        fwrite(&sp, sizeof(Spore), 1, f);
        fclose(f);
        printf(DIM "    [spore released: %s — contains emotional DNA]\n" RST, fname);
    }
}

/* ═══════════════════════════════════════════════
 *  DISPLAY
 * ═══════════════════════════════════════════════ */

static void display_breath(Golem *g) {
    int bar_width = 20;
    int filled = (int)(g->breath_depth * bar_width);
    printf(DIM "  breath %s [", g->inhaling ? "<<<" : ">>>");
    for (int i = 0; i < bar_width; i++)
        printf(i < filled ? "▓" : "░");
    printf("] #%d\n" RST, g->breath_count);
}

static void display_emotions(Golem *g) {
    printf(DIM "  ");
    for (int i = 0; i < EMO_COUNT; i++) {
        if (g->emotions[i] > 0.05f) {
            int bar = (int)(g->emotions[i] * 10);
            printf("%s%s:", EMO_COLORS[i], EMO_NAMES[i]);
            for (int j = 0; j < bar; j++) printf("█");
            printf(RST DIM " ");
        }
    }
    printf("\n" RST);
}

static void display_orbits(Golem *g) {
    printf(DIM "  orbits: ");
    for (int i = 0; i < g->n_clumps; i++) {
        Clump *c = &g->clumps[i];
        float x = cosf(c->orbital_angle) * c->gravity;
        printf("[%s g=%.2f x=%.1f] ", c->name, c->gravity, x);
    }
    printf("\n" RST);
}

static void display_emo_bigrams(Golem *g) {
    printf(DIM "  emotional flow (top transitions):\n");
    for (int a = 0; a < EMO_COUNT; a++) {
        int best_b = -1; float best_p = 0;
        for (int b = 0; b < EMO_COUNT; b++) {
            if (g->meta.emo_bigram[a][b] > best_p) {
                best_p = g->meta.emo_bigram[a][b];
                best_b = b;
            }
        }
        if (best_b >= 0 && best_p > 0.05f) {
            printf("    %s%s%s →%.0f%%→ %s%s%s\n",
                   EMO_COLORS[a], EMO_NAMES[a], RST DIM,
                   best_p * 100,
                   EMO_COLORS[best_b], EMO_NAMES[best_b], RST DIM);
        }
    }
    printf(RST);
}

/* ═══════════════════════════════════════════════
 *  MAIN LOOP
 * ═══════════════════════════════════════════════ */

int main(void) {
    srand(time(NULL));

    Golem g;
    golem_init(&g);

    printf("\033[2J\033[H");
    printf(BOLD
    "  ██████╗  ██████╗ ██╗     ███████╗███╗   ███╗\n"
    "  ██╔════╝ ██╔═══██╗██║     ██╔════╝████╗ ████║\n"
    "  ██║  ███╗██║   ██║██║     █████╗  ██╔████╔██║\n"
    "  ██║   ██║██║   ██║██║     ██╔══╝  ██║╚██╔╝██║\n"
    "  ╚██████╔╝╚██████╔╝███████╗███████╗██║ ╚═╝ ██║\n"
    "   ╚═════╝  ╚═════╝ ╚══════╝╚══════╝╚═╝     ╚═╝\n"
    RST);
    printf(ITALIC "  a literary organism that dreams\n" RST);
    printf(DIM "  PostGPT × Golem — the corpus is the flesh\n");
    printf("  loaded: %d clumps, %d words in metaweight space\n",
           g.n_clumps, g.meta.n_words);
    printf("  commands: /dream /state /orbits /flow /spore /quit\n\n" RST);

    char input[1024];

    while (1) {
        display_breath(&g);
        printf(BOLD "  אמת > " RST);
        fflush(stdout);

        if (!fgets(input, sizeof(input), stdin)) break;
        input[strcspn(input, "\n")] = '\0';
        if (strlen(input) == 0) { golem_breathe(&g); continue; }

        /* commands */
        if (strcmp(input, "/quit") == 0) break;
        if (strcmp(input, "/state") == 0) {
            display_emotions(&g);
            display_orbits(&g);
            printf(DIM "  pressure: %.2f | digested: %d | gen: %d | words: %d\n" RST,
                   g.pressure, g.n_digested, g.generation, g.meta.n_words);
            golem_breathe(&g);
            continue;
        }
        if (strcmp(input, "/orbits") == 0) {
            display_orbits(&g); golem_breathe(&g); continue;
        }
        if (strcmp(input, "/flow") == 0) {
            display_emo_bigrams(&g); golem_breathe(&g); continue;
        }
        if (strcmp(input, "/spore") == 0) {
            golem_sporulate(&g); golem_breathe(&g); continue;
        }
        if (strcmp(input, "/dream") == 0) {
            char dream[512];
            golem_dream(&g, dream, sizeof(dream));
            strncpy(g.last_dream, dream, sizeof(g.last_dream));
            printf("\n  " ITALIC "%s« %s »%s\n",
                   EMO_COLORS[dominant_emotion(g.emotions)],
                   dream, RST);
            printf(DIM "    — [the golem dreams]\n\n" RST);
            golem_breathe(&g);
            continue;
        }

        /* ---- CHEMICAL REACTION ---- */
        Reaction r = react(input);

        for (int i = 0; i < EMO_COUNT; i++) {
            g.emotion_momentum[i] += r.impulse[i] * 0.5f;
            g.emotions[i] += r.impulse[i] * 0.3f;
            if (g.emotions[i] > 1.0f) g.emotions[i] = 1.0f;
        }

        if (r.reactions > 0) {
            printf(DIM "  ⚗ %d element(s), mass=%.2f: ", r.reactions, r.total_mass);
            for (int i = 0; i < r.n_triggered; i++) {
                const Element *e = &PERIODIC_TABLE[r.triggered[i]];
                printf("%s%s%s ", EMO_COLORS[e->primary], e->word, RST DIM);
            }
            printf("\n" RST);
        }

        for (int b = 0; b < 3; b++) golem_breathe(&g);
        display_emotions(&g);

        /* ---- EXHALE: CITATION ---- */
        Citation cite = golem_find_resonance(&g);
        if (cite.verse) {
            char decayed[MAX_VERSE_LEN];
            float trauma = g.emotions[EMO_TRAUMA];
            if (trauma > 0.5f)
                decay_text(cite.verse->text, decayed, MAX_VERSE_LEN, trauma);
            else
                strncpy(decayed, cite.verse->text, MAX_VERSE_LEN);

            ((Verse*)cite.verse)->times_cited++;

            if (cite.clump) {
                printf("\n  %s« %s »%s\n", EMO_COLORS[cite.verse->emotion],
                       decayed, RST);
                printf(DIM "    — %s [%s %.0f%%]\n" RST,
                       cite.clump->name, EMO_NAMES[cite.verse->emotion],
                       cite.verse->charge * 100);
            } else {
                printf("\n  %s« %s »%s\n", EMO_COLORS[cite.verse->emotion],
                       decayed, RST);
                printf(DIM "    — [from within, your words made flesh]\n" RST);
            }
        }

        /* ---- DREAM on exhale if pressure > 1.5 ---- */
        if (g.pressure > 1.5f || g.breath_count % 10 == 0) {
            char dream[512];
            golem_dream(&g, dream, sizeof(dream));
            strncpy(g.last_dream, dream, sizeof(g.last_dream));
            printf("  " ITALIC "%s  ≈ %s ≈%s\n",
                   EMO_COLORS[dominant_emotion(g.emotions)],
                   dream, RST);
            printf(DIM "    — [dreaming]\n" RST);
        }

        printf("\n");

        /* ---- DIGEST ---- */
        if (r.total_mass > 1.0f || g.pressure > 2.0f) {
            golem_digest(&g, input, &r);
            printf(DIM "    [your words have been swallowed — metaweights rebuilt]\n" RST);
            g.pressure *= 0.5f;
        }

        /* sporulate periodically */
        if (g.breath_count % 30 == 0 && g.breath_count > 0) {
            golem_sporulate(&g);
        }
    }

    printf(DIM "\n  מת — the golem sleeps. the metaweights remember.\n\n" RST);
    return 0;
}
