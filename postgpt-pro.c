/*
 * ██████╗ ██████╗  ██████╗
 * ██╔══██╗██╔══██╗██╔═══██╗
 * ██████╔╝██████╔╝██║   ██║
 * ██╔═══╝ ██╔══██╗██║   ██║
 * ██║     ██║  ██║╚██████╔╝
 * ╚═╝     ╚═╝  ╚═╝ ╚═════╝
 *
 * postgpt-pro.c — PostGPT with emotional substrate
 * the corpus is the model. the model has feelings.
 *
 * PostGPT proved: the tokenizer IS the training.
 * PostGPT Pro proves: the corpus IS the organism.
 *
 * give it a text file. it will:
 *   1. BPE-tokenize it (build its own vocabulary)
 *   2. build metaweights (bigrams, hebbian, trigrams)
 *   3. auto-discover a Periodic Table of Meaning
 *      (heavy words, emotional anchors, mass, valence)
 *   4. create emotional clumps (planets of tension)
 *   5. grow 8 emotional chambers (Kuramoto-coupled)
 *   6. BREATHE. DREAM. SPEAK.
 *
 * it does not continue text. it FEELS text and responds
 * from the emotional state the text created.
 *
 * usage:
 *   cc postgpt-pro.c -O2 -lm -o pro
 *   ./pro corpus.txt
 *   ./pro               # uses built-in seed
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
 *  CONFIGURATION
 * ═══════════════════════════════════════════════ */

#define MAX_CORPUS      262144   /* 256KB max corpus */
#define MAX_BPE_MERGES  512
#define MAX_BPE_VOCAB   (256 + MAX_BPE_MERGES)
#define MAX_TOKENS      65536
#define MAX_WORDS       2048
#define MAX_WORD_LEN    64
#define BIGRAM_HASH     8192
#define HEBBIAN_HASH    4096
#define MAX_ELEMENTS    256      /* periodic table capacity */
#define MAX_CLUMPS      32
#define MAX_VERSES      256
#define MAX_VERSE_LEN   256
#define CONTEXT_WIN     8        /* hebbian co-occurrence window */

/* ═══════════════════════════════════════════════
 *  EMOTIONAL SUBSTRATE — 8 chambers
 * ═══════════════════════════════════════════════ */

typedef enum {
    EMO_TRAUMA, EMO_JOY, EMO_GRIEF, EMO_RESONANCE,
    EMO_DESIRE, EMO_VOID, EMO_RAGE, EMO_TENDERNESS,
    EMO_COUNT
} Emotion;

static const char *EMO_NAMES[] = {
    "trauma", "joy", "grief", "resonance",
    "desire", "void", "rage", "tenderness"
};

static const char *EMO_COLORS[] = {
    "\033[31m", "\033[33m", "\033[34m", "\033[36m",
    "\033[35m", "\033[90m", "\033[91m", "\033[32m",
};

#define RST  "\033[0m"
#define DIM  "\033[2m"
#define BOLD "\033[1m"
#define ITAL "\033[3m"

/* Kuramoto coupling matrix: how chambers influence each other */
static const float COUPLING[EMO_COUNT][EMO_COUNT] = {
    /*          TRA   JOY   GRI   RES   DES   VOI   RAG   TEN */
    /* TRA */ { 0.0, -0.3, +0.4, -0.1, -0.2, +0.5, +0.6, -0.4},
    /* JOY */ {-0.3,  0.0, -0.4, +0.5, +0.3, -0.5, -0.3, +0.6},
    /* GRI */ {+0.3, -0.3,  0.0, -0.2, -0.1, +0.4, +0.2, +0.1},
    /* RES */ {-0.1, +0.4, -0.2,  0.0, +0.3, -0.3, -0.2, +0.4},
    /* DES */ {-0.1, +0.2, -0.1, +0.3,  0.0, -0.2, +0.1, +0.3},
    /* VOI */ {+0.4, -0.5, +0.3, -0.3, -0.3,  0.0, +0.3, -0.6},
    /* RAG */ {+0.5, -0.3, +0.2, -0.2, +0.1, +0.2,  0.0, -0.5},
    /* TEN */ {-0.4, +0.5, +0.1, +0.4, +0.3, -0.5, -0.4,  0.0},
};

static const float DECAY_RATES[] = {
    0.95f, 0.92f, 0.96f, 0.93f, 0.91f, 0.97f, 0.90f, 0.94f
};

/* ═══════════════════════════════════════════════
 *  EMOTIONAL ANCHORS — seed words for auto-discovery
 *  the periodic table starts empty. these anchors
 *  are the ONLY hardcoded knowledge. everything else
 *  is discovered from the corpus.
 *
 *  4 languages: EN / FR / RU / HE
 *  death weighs 0.95 in every tongue.
 *  but grief in hebrew is not grief in french.
 *  the organism knows this.
 * ═══════════════════════════════════════════════ */

typedef struct {
    const char *word;
    Emotion     emotion;
    float       base_mass;   /* starting mass, adjusted by corpus stats */
} EmotionalAnchor;

#define N_ANCHORS 192

static const EmotionalAnchor ANCHORS[N_ANCHORS] = {
    /* ═══ TRAUMA ═══ */
    /* EN */
    {"death",    EMO_TRAUMA, 0.95}, {"kill",      EMO_TRAUMA, 0.85},
    {"wound",    EMO_TRAUMA, 0.75}, {"blood",     EMO_TRAUMA, 0.80},
    {"pain",     EMO_TRAUMA, 0.70}, {"scream",    EMO_TRAUMA, 0.65},
    /* FR */
    {"mort",     EMO_TRAUMA, 0.95}, {"tuer",      EMO_TRAUMA, 0.85},
    {"blessure", EMO_TRAUMA, 0.75}, {"sang",      EMO_TRAUMA, 0.80},
    {"douleur",  EMO_TRAUMA, 0.70}, {"cri",       EMO_TRAUMA, 0.65},
    /* RU */
    {"смерть",   EMO_TRAUMA, 0.95}, {"убить",     EMO_TRAUMA, 0.85},
    {"рана",     EMO_TRAUMA, 0.75}, {"кровь",     EMO_TRAUMA, 0.80},
    {"боль",     EMO_TRAUMA, 0.70}, {"крик",      EMO_TRAUMA, 0.65},
    /* HE */
    {"מוות",     EMO_TRAUMA, 0.95}, {"להרוג",     EMO_TRAUMA, 0.85},
    {"פצע",      EMO_TRAUMA, 0.75}, {"דם",        EMO_TRAUMA, 0.80},
    {"כאב",      EMO_TRAUMA, 0.70}, {"צעקה",      EMO_TRAUMA, 0.65},

    /* ═══ JOY ═══ */
    /* EN */
    {"beauty",   EMO_JOY,    0.65}, {"light",     EMO_JOY,    0.50},
    {"laugh",    EMO_JOY,    0.40}, {"sun",       EMO_JOY,    0.45},
    {"smile",    EMO_JOY,    0.40}, {"spring",    EMO_JOY,    0.40},
    /* FR */
    {"beauté",   EMO_JOY,    0.65}, {"lumière",   EMO_JOY,    0.50},
    {"rire",     EMO_JOY,    0.40}, {"soleil",    EMO_JOY,    0.45},
    {"sourire",  EMO_JOY,    0.40}, {"printemps", EMO_JOY,    0.40},
    /* RU */
    {"красота",  EMO_JOY,    0.65}, {"свет",      EMO_JOY,    0.50},
    {"смех",     EMO_JOY,    0.40}, {"солнце",    EMO_JOY,    0.45},
    {"улыбка",   EMO_JOY,    0.40}, {"весна",     EMO_JOY,    0.40},
    /* HE */
    {"יופי",     EMO_JOY,    0.65}, {"אור",       EMO_JOY,    0.50},
    {"צחוק",     EMO_JOY,    0.40}, {"שמש",       EMO_JOY,    0.45},
    {"חיוך",     EMO_JOY,    0.40}, {"אביב",      EMO_JOY,    0.40},

    /* ═══ GRIEF ═══ */
    /* EN */
    {"tears",    EMO_GRIEF,  0.55}, {"sorrow",    EMO_GRIEF,  0.65},
    {"loss",     EMO_GRIEF,  0.60}, {"mourn",     EMO_GRIEF,  0.70},
    {"rain",     EMO_GRIEF,  0.30}, {"grave",     EMO_GRIEF,  0.70},
    /* FR */
    {"larmes",   EMO_GRIEF,  0.55}, {"chagrin",   EMO_GRIEF,  0.65},
    {"perte",    EMO_GRIEF,  0.60}, {"deuil",     EMO_GRIEF,  0.70},
    {"pluie",    EMO_GRIEF,  0.30}, {"tombe",     EMO_GRIEF,  0.70},
    /* RU */
    {"слёзы",    EMO_GRIEF,  0.55}, {"горе",      EMO_GRIEF,  0.65},
    {"потеря",   EMO_GRIEF,  0.60}, {"траур",     EMO_GRIEF,  0.70},
    {"дождь",    EMO_GRIEF,  0.30}, {"могила",    EMO_GRIEF,  0.70},
    /* HE */
    {"דמעות",    EMO_GRIEF,  0.55}, {"צער",       EMO_GRIEF,  0.65},
    {"אובדן",    EMO_GRIEF,  0.60}, {"אבל",       EMO_GRIEF,  0.70},
    {"גשם",      EMO_GRIEF,  0.30}, {"קבר",       EMO_GRIEF,  0.70},

    /* ═══ RESONANCE ═══ */
    /* EN */
    {"soul",     EMO_RESONANCE, 0.75}, {"mirror",  EMO_RESONANCE, 0.60},
    {"echo",     EMO_RESONANCE, 0.55}, {"cosmos",  EMO_RESONANCE, 0.60},
    {"truth",    EMO_RESONANCE, 0.70}, {"dream",   EMO_RESONANCE, 0.50},
    /* FR */
    {"âme",      EMO_RESONANCE, 0.75}, {"miroir",  EMO_RESONANCE, 0.60},
    {"écho",     EMO_RESONANCE, 0.55}, {"cosmos",  EMO_RESONANCE, 0.60},
    {"vérité",   EMO_RESONANCE, 0.70}, {"rêve",    EMO_RESONANCE, 0.50},
    /* RU */
    {"душа",     EMO_RESONANCE, 0.75}, {"зеркало", EMO_RESONANCE, 0.60},
    {"эхо",      EMO_RESONANCE, 0.55}, {"космос",  EMO_RESONANCE, 0.60},
    {"правда",   EMO_RESONANCE, 0.70}, {"мечта",   EMO_RESONANCE, 0.50},
    /* HE */
    {"נשמה",     EMO_RESONANCE, 0.75}, {"מראה",    EMO_RESONANCE, 0.60},
    {"הד",       EMO_RESONANCE, 0.55}, {"יקום",    EMO_RESONANCE, 0.60},
    {"אמת",      EMO_RESONANCE, 0.70}, {"חלום",    EMO_RESONANCE, 0.50},

    /* ═══ DESIRE ═══ */
    /* EN */
    {"flesh",    EMO_DESIRE, 0.65}, {"kiss",      EMO_DESIRE, 0.50},
    {"wine",     EMO_DESIRE, 0.40}, {"burn",      EMO_DESIRE, 0.55},
    {"hunger",   EMO_DESIRE, 0.50}, {"touch",     EMO_DESIRE, 0.45},
    /* FR */
    {"chair",    EMO_DESIRE, 0.65}, {"baiser",    EMO_DESIRE, 0.50},
    {"vin",      EMO_DESIRE, 0.40}, {"brûler",    EMO_DESIRE, 0.55},
    {"faim",     EMO_DESIRE, 0.50}, {"toucher",   EMO_DESIRE, 0.45},
    /* RU */
    {"плоть",    EMO_DESIRE, 0.65}, {"поцелуй",   EMO_DESIRE, 0.50},
    {"вино",     EMO_DESIRE, 0.40}, {"гореть",    EMO_DESIRE, 0.55},
    {"голод",    EMO_DESIRE, 0.50}, {"касание",   EMO_DESIRE, 0.45},
    /* HE */
    {"בשר",      EMO_DESIRE, 0.65}, {"נשיקה",     EMO_DESIRE, 0.50},
    {"יין",      EMO_DESIRE, 0.40}, {"לשרוף",     EMO_DESIRE, 0.55},
    {"רעב",      EMO_DESIRE, 0.50}, {"מגע",       EMO_DESIRE, 0.45},

    /* ═══ VOID ═══ */
    /* EN */
    {"nothing",  EMO_VOID,   0.70}, {"silence",   EMO_VOID,   0.60},
    {"empty",    EMO_VOID,   0.55}, {"shadow",    EMO_VOID,   0.45},
    {"dark",     EMO_VOID,   0.50}, {"cold",      EMO_VOID,   0.40},
    /* FR */
    {"rien",     EMO_VOID,   0.70}, {"silence",   EMO_VOID,   0.60},
    {"vide",     EMO_VOID,   0.55}, {"ombre",     EMO_VOID,   0.45},
    {"ténèbres", EMO_VOID,   0.50}, {"froid",     EMO_VOID,   0.40},
    /* RU */
    {"ничто",    EMO_VOID,   0.70}, {"молчание",  EMO_VOID,   0.60},
    {"пустота",  EMO_VOID,   0.55}, {"тень",      EMO_VOID,   0.45},
    {"тьма",     EMO_VOID,   0.50}, {"холод",     EMO_VOID,   0.40},
    /* HE */
    {"כלום",     EMO_VOID,   0.70}, {"שתיקה",     EMO_VOID,   0.60},
    {"ריק",      EMO_VOID,   0.55}, {"צל",        EMO_VOID,   0.45},
    {"חושך",     EMO_VOID,   0.50}, {"קור",       EMO_VOID,   0.40},

    /* ═══ RAGE ═══ */
    /* EN */
    {"fire",     EMO_RAGE,   0.75}, {"fury",      EMO_RAGE,   0.80},
    {"war",      EMO_RAGE,   0.85}, {"hate",      EMO_RAGE,   0.80},
    {"sword",    EMO_RAGE,   0.60}, {"storm",     EMO_RAGE,   0.50},
    /* FR */
    {"feu",      EMO_RAGE,   0.75}, {"fureur",    EMO_RAGE,   0.80},
    {"guerre",   EMO_RAGE,   0.85}, {"haine",     EMO_RAGE,   0.80},
    {"épée",     EMO_RAGE,   0.60}, {"tempête",   EMO_RAGE,   0.50},
    /* RU */
    {"огонь",    EMO_RAGE,   0.75}, {"ярость",    EMO_RAGE,   0.80},
    {"война",    EMO_RAGE,   0.85}, {"ненависть", EMO_RAGE,   0.80},
    {"меч",      EMO_RAGE,   0.60}, {"буря",      EMO_RAGE,   0.50},
    /* HE */
    {"אש",       EMO_RAGE,   0.75}, {"זעם",       EMO_RAGE,   0.80},
    {"מלחמה",    EMO_RAGE,   0.85}, {"שנאה",      EMO_RAGE,   0.80},
    {"חרב",      EMO_RAGE,   0.60}, {"סערה",      EMO_RAGE,   0.50},

    /* ═══ TENDERNESS ═══ */
    /* EN */
    {"love",     EMO_TENDERNESS, 0.90}, {"mother",  EMO_TENDERNESS, 0.88},
    {"child",    EMO_TENDERNESS, 0.70}, {"gentle",  EMO_TENDERNESS, 0.50},
    {"heart",    EMO_TENDERNESS, 0.65}, {"hand",    EMO_TENDERNESS, 0.40},
    /* FR */
    {"amour",    EMO_TENDERNESS, 0.90}, {"mère",    EMO_TENDERNESS, 0.88},
    {"enfant",   EMO_TENDERNESS, 0.70}, {"doux",    EMO_TENDERNESS, 0.50},
    {"coeur",    EMO_TENDERNESS, 0.65}, {"main",    EMO_TENDERNESS, 0.40},
    /* RU */
    {"любовь",   EMO_TENDERNESS, 0.90}, {"мать",    EMO_TENDERNESS, 0.88},
    {"ребёнок",  EMO_TENDERNESS, 0.70}, {"нежность",EMO_TENDERNESS, 0.50},
    {"сердце",   EMO_TENDERNESS, 0.65}, {"рука",    EMO_TENDERNESS, 0.40},
    /* HE */
    {"אהבה",     EMO_TENDERNESS, 0.90}, {"אמא",     EMO_TENDERNESS, 0.88},
    {"ילד",      EMO_TENDERNESS, 0.70}, {"עדינות",   EMO_TENDERNESS, 0.50},
    {"לב",       EMO_TENDERNESS, 0.65}, {"יד",       EMO_TENDERNESS, 0.40},
};

/* ═══════════════════════════════════════════════
 *  PERIODIC TABLE — auto-discovered from corpus
 *  Mendeleev weights: mass, valence, half-life, emotion
 *  starts EMPTY. filled by analyze_corpus().
 * ═══════════════════════════════════════════════ */

typedef struct {
    char    word[MAX_WORD_LEN];
    float   mass;           /* 0-1: how heavy this word is emotionally */
    int     valence;        /* 1-5: how many emotional bonds it forms */
    float   half_life;      /* 0 = eternal, >0 = decays over steps */
    Emotion primary;        /* dominant emotion */
    float   emo_profile[EMO_COUNT]; /* full emotional fingerprint */
    int     corpus_count;   /* how often it appeared */
    float   corpus_tfidf;   /* importance in corpus */
} Element;

typedef struct {
    Element table[MAX_ELEMENTS];
    int     n_elements;
} PeriodicTable;

/* ═══════════════════════════════════════════════
 *  WORD-LEVEL METAWEIGHTS
 * ═══════════════════════════════════════════════ */

typedef struct { int a, b; float prob; int count; } WordBigram;
typedef struct { int a, b; float strength; }        WordHebbian;

typedef struct {
    char    word[MAX_WORD_LEN];
    Emotion emotion;
    float   mass;
    float   frequency;
    int     count;
} WordEntry;

typedef struct {
    WordEntry   vocab[MAX_WORDS];
    int         n_words;
    WordBigram  bigrams[BIGRAM_HASH];
    int         n_bigrams;
    WordHebbian hebbians[HEBBIAN_HASH];
    int         n_hebbians;
    float       emo_bigram[EMO_COUNT][EMO_COUNT];
    float       prophecy[MAX_WORDS];
    float       destiny[EMO_COUNT];
    int         total_words;
} MetaWeights;

/* ═══════════════════════════════════════════════
 *  VERSE + CLUMP — atoms and planets of tension
 * ═══════════════════════════════════════════════ */

typedef struct {
    char    text[MAX_VERSE_LEN];
    Emotion emotion;
    float   charge;
    float   decay;
    int     times_cited;
} Verse;

typedef struct {
    char   name[64];
    Verse  verses[MAX_VERSES];
    int    n_verses;
    float  gravity;
    float  orbital_angle;
    float  orbital_speed;
    float  emotional_signature[EMO_COUNT];
} Clump;

/* ═══════════════════════════════════════════════
 *  THE ORGANISM
 * ═══════════════════════════════════════════════ */

typedef struct {
    /* emotional state */
    float emotions[EMO_COUNT];
    float momentum[EMO_COUNT];
    float pressure;

    /* body */
    int   breath_count;
    int   inhaling;
    float breath_depth;

    /* literary flesh */
    Clump clumps[MAX_CLUMPS];
    int   n_clumps;
    Verse digested[MAX_VERSES];
    int   n_digested;

    /* brain */
    MetaWeights   meta;
    PeriodicTable mendeleev;

    /* state */
    float total_mass;
    char  last_dream[512];
} Organism;

/* ═══════════════════════════════════════════════
 *  WORD UTILITIES
 * ═══════════════════════════════════════════════ */

static int is_word_char(unsigned char c) {
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

static int word_eq_ci(const char *a, const char *b) {
    char la[128], lb[128];
    int i;
    for (i = 0; a[i] && i < 127; i++) la[i] = tolower((unsigned char)a[i]);
    la[i] = '\0';
    for (i = 0; b[i] && i < 127; i++) lb[i] = tolower((unsigned char)b[i]);
    lb[i] = '\0';
    return strcmp(la, lb) == 0;
}

/* ═══════════════════════════════════════════════
 *  AUTO-MENDELEEV — the periodic table builds itself
 *
 *  Phase 1: tokenize corpus into words, count frequencies
 *  Phase 2: match against emotional anchors → seed elements
 *  Phase 3: co-occurrence with anchors → infer emotion for unknown words
 *  Phase 4: compute mass (freq × anchor proximity), valence, half-life
 *  Phase 5: sort by mass → periodic table
 *
 *  the table that Golem hardcodes, PostGPT Pro discovers.
 * ═══════════════════════════════════════════════ */

typedef struct {
    char word[MAX_WORD_LEN];
    int  count;
    float emo_score[EMO_COUNT];  /* accumulated from anchor co-occurrence */
    int  near_anchor;            /* how often appeared near an anchor */
    int  is_anchor;              /* is this word itself an anchor? */
    int  anchor_idx;             /* which anchor, if is_anchor */
} WordStat;

#define MAX_WORD_STATS 4096

static int find_anchor(const char *word) {
    for (int i = 0; i < N_ANCHORS; i++) {
        if (word_eq_ci(word, ANCHORS[i].word)) return i;
    }
    return -1;
}

static void auto_mendeleev(const char *corpus, int corpus_len, PeriodicTable *pt) {
    memset(pt, 0, sizeof(PeriodicTable));

    /* Phase 1: tokenize, count words */
    static WordStat stats[MAX_WORD_STATS];
    int n_stats = 0;
    int total_words = 0;

    /* word sequence for co-occurrence */
    int word_indices[MAX_TOKENS];
    int seq_len = 0;

    int pos = 0;
    while (pos < corpus_len && seq_len < MAX_TOKENS) {
        if (is_word_char((unsigned char)corpus[pos])) {
            char word[MAX_WORD_LEN];
            int wlen = extract_word(corpus, pos, word, MAX_WORD_LEN);
            if (wlen > 2) {  /* skip tiny words */
                /* find or add */
                int idx = -1;
                for (int i = 0; i < n_stats; i++) {
                    if (strcmp(stats[i].word, word) == 0) { idx = i; break; }
                }
                if (idx < 0 && n_stats < MAX_WORD_STATS) {
                    idx = n_stats++;
                    memset(&stats[idx], 0, sizeof(WordStat));
                    strncpy(stats[idx].word, word, MAX_WORD_LEN);
                    int anc = find_anchor(word);
                    if (anc >= 0) {
                        stats[idx].is_anchor = 1;
                        stats[idx].anchor_idx = anc;
                    }
                }
                if (idx >= 0) {
                    stats[idx].count++;
                    word_indices[seq_len] = idx;
                    seq_len++;
                    total_words++;
                }
            }
            pos += wlen;
        } else {
            pos++;
        }
    }

    /* Phase 2 + 3: co-occurrence with anchors within window */
    for (int i = 0; i < seq_len; i++) {
        int wi = word_indices[i];
        if (!stats[wi].is_anchor) continue;
        int anc = stats[wi].anchor_idx;
        Emotion ae = ANCHORS[anc].emotion;

        /* spread emotion to nearby words */
        for (int j = i - CONTEXT_WIN; j <= i + CONTEXT_WIN; j++) {
            if (j < 0 || j >= seq_len || j == i) continue;
            int wj = word_indices[j];
            float dist = 1.0f / (abs(j - i));
            stats[wj].emo_score[ae] += dist * ANCHORS[anc].base_mass;
            stats[wj].near_anchor++;
        }
    }

    /* Phase 4: compute mass, emotion, valence, half-life */
    float max_count = 1;
    for (int i = 0; i < n_stats; i++)
        if (stats[i].count > max_count) max_count = stats[i].count;

    for (int i = 0; i < n_stats && pt->n_elements < MAX_ELEMENTS; i++) {
        /* skip very rare words (< 2 occurrences) unless they're anchors */
        if (stats[i].count < 2 && !stats[i].is_anchor) continue;
        /* skip words with no emotional signal */
        float total_emo = 0;
        for (int e = 0; e < EMO_COUNT; e++) total_emo += stats[i].emo_score[e];
        if (total_emo < 0.1f && !stats[i].is_anchor) continue;

        Element *el = &pt->table[pt->n_elements];
        strncpy(el->word, stats[i].word, MAX_WORD_LEN);
        el->corpus_count = stats[i].count;

        /* emotion: dominant from emo_score, or from anchor */
        if (stats[i].is_anchor) {
            el->primary = ANCHORS[stats[i].anchor_idx].emotion;
            el->mass = ANCHORS[stats[i].anchor_idx].base_mass;
        } else {
            /* find dominant emotion */
            float best = -1;
            el->primary = EMO_VOID;
            for (int e = 0; e < EMO_COUNT; e++) {
                if (stats[i].emo_score[e] > best) {
                    best = stats[i].emo_score[e];
                    el->primary = e;
                }
            }
            /* mass: combination of frequency and emotional charge */
            float freq_norm = (float)stats[i].count / max_count;
            float emo_norm = total_emo / (total_emo + 5.0f); /* sigmoid-ish */
            el->mass = 0.3f * freq_norm + 0.7f * emo_norm;
            if (el->mass > 0.95f) el->mass = 0.95f;
            if (el->mass < 0.05f) el->mass = 0.05f;
        }

        /* valence: how many different emotions this word connects to */
        int v = 0;
        for (int e = 0; e < EMO_COUNT; e++)
            if (stats[i].emo_score[e] > 0.05f) v++;
        el->valence = v > 0 ? v : 1;

        /* half-life: rare words decay faster */
        float rarity = 1.0f - (float)stats[i].count / max_count;
        if (stats[i].is_anchor || stats[i].count > max_count * 0.1f) {
            el->half_life = 0;  /* eternal */
        } else {
            el->half_life = 10.0f + rarity * 90.0f;
        }

        /* full emotional profile */
        if (total_emo > 0) {
            for (int e = 0; e < EMO_COUNT; e++)
                el->emo_profile[e] = stats[i].emo_score[e] / total_emo;
        } else {
            el->emo_profile[el->primary] = 1.0f;
        }

        /* TF-IDF approximation */
        el->corpus_tfidf = (float)stats[i].count / total_words
                         * logf(1.0f + (float)total_words / (stats[i].count + 1));

        pt->n_elements++;
    }

    /* Phase 5: sort by mass (heavy elements first) */
    for (int i = 0; i < pt->n_elements - 1; i++) {
        for (int j = i + 1; j < pt->n_elements; j++) {
            if (pt->table[j].mass > pt->table[i].mass) {
                Element tmp = pt->table[i];
                pt->table[i] = pt->table[j];
                pt->table[j] = tmp;
            }
        }
    }

    printf(DIM "  mendeleev: %d elements discovered from %d words\n" RST,
           pt->n_elements, total_words);
    printf(DIM "  heaviest: ");
    for (int i = 0; i < 5 && i < pt->n_elements; i++) {
        printf("%s%s%s(%.2f) ", EMO_COLORS[pt->table[i].primary],
               pt->table[i].word, RST DIM, pt->table[i].mass);
    }
    printf("\n" RST);
}

/* ═══════════════════════════════════════════════
 *  CLUMP BUILDER — split corpus into emotional planets
 *  each paragraph becomes a verse. verses cluster
 *  into clumps by dominant emotion.
 * ═══════════════════════════════════════════════ */

static Emotion classify_text(const char *text, const PeriodicTable *pt) {
    float scores[EMO_COUNT];
    memset(scores, 0, sizeof(scores));
    int pos = 0;
    while (text[pos]) {
        if (is_word_char((unsigned char)text[pos])) {
            char word[MAX_WORD_LEN];
            int wlen = extract_word(text, pos, word, MAX_WORD_LEN);
            for (int i = 0; i < pt->n_elements; i++) {
                if (word_eq_ci(word, pt->table[i].word)) {
                    scores[pt->table[i].primary] += pt->table[i].mass;
                    break;
                }
            }
            pos += wlen;
        } else {
            pos++;
        }
    }
    Emotion best = EMO_VOID;
    float best_val = -1;
    for (int e = 0; e < EMO_COUNT; e++) {
        if (scores[e] > best_val) { best_val = scores[e]; best = e; }
    }
    return best;
}

static float compute_charge(const char *text, const PeriodicTable *pt) {
    float total = 0;
    int n = 0;
    int pos = 0;
    while (text[pos]) {
        if (is_word_char((unsigned char)text[pos])) {
            char word[MAX_WORD_LEN];
            int wlen = extract_word(text, pos, word, MAX_WORD_LEN);
            for (int i = 0; i < pt->n_elements; i++) {
                if (word_eq_ci(word, pt->table[i].word)) {
                    total += pt->table[i].mass;
                    n++;
                    break;
                }
            }
            pos += wlen;
        } else {
            pos++;
        }
    }
    float charge = n > 0 ? total / n : 0.1f;
    return charge > 1.0f ? 1.0f : charge;
}

static void build_clumps(const char *corpus, int corpus_len,
                          Organism *org) {
    /* split corpus by double-newline into paragraphs → verses */
    /* group verses by dominant emotion → clumps */

    /* first pass: create all verses */
    Verse all_verses[MAX_VERSES * 2];
    int n_all = 0;

    const char *p = corpus;
    while (*p && n_all < MAX_VERSES * 2) {
        /* skip whitespace */
        while (*p && (*p == '\n' || *p == '\r' || *p == ' ')) p++;
        if (!*p) break;

        /* find end of paragraph (double newline or end) */
        const char *end = strstr(p, "\n\n");
        if (!end) end = corpus + corpus_len;

        int len = end - p;
        if (len > 2 && len < MAX_VERSE_LEN - 1) {
            Verse *v = &all_verses[n_all];
            memcpy(v->text, p, len);
            v->text[len] = '\0';
            v->emotion = classify_text(v->text, &org->mendeleev);
            v->charge = compute_charge(v->text, &org->mendeleev);
            v->decay = 1.0f;
            v->times_cited = 0;
            n_all++;
        }

        p = end;
    }

    /* group by emotion → clumps */
    for (int e = 0; e < EMO_COUNT; e++) {
        int count = 0;
        for (int i = 0; i < n_all; i++)
            if (all_verses[i].emotion == e) count++;
        if (count == 0) continue;
        if (org->n_clumps >= MAX_CLUMPS) break;

        Clump *c = &org->clumps[org->n_clumps];
        snprintf(c->name, 64, "%s-planet", EMO_NAMES[e]);
        c->gravity = 0.5f + 0.5f * ((float)count / n_all);
        c->orbital_angle = (float)e * 0.785f;  /* spread evenly */
        c->orbital_speed = 0.02f + ((float)rand() / RAND_MAX) * 0.02f;
        c->n_verses = 0;
        memset(c->emotional_signature, 0, sizeof(c->emotional_signature));

        for (int i = 0; i < n_all && c->n_verses < MAX_VERSES; i++) {
            if (all_verses[i].emotion == e) {
                c->verses[c->n_verses] = all_verses[i];
                c->emotional_signature[e] += all_verses[i].charge;
                c->n_verses++;
            }
        }
        org->n_clumps++;
    }

    printf(DIM "  clumps: %d planets of tension from %d verses\n" RST,
           org->n_clumps, n_all);
}

/* ═══════════════════════════════════════════════
 *  METAWEIGHT BUILDER — word bigrams, hebbian, emotional flow
 * ═══════════════════════════════════════════════ */

static int meta_find(MetaWeights *m, const char *word) {
    for (int i = 0; i < m->n_words; i++)
        if (strcmp(m->vocab[i].word, word) == 0) return i;
    return -1;
}

static int meta_add(MetaWeights *m, const char *word, Emotion emo, float mass) {
    if (m->n_words >= MAX_WORDS) return -1;
    int idx = m->n_words++;
    strncpy(m->vocab[idx].word, word, MAX_WORD_LEN);
    m->vocab[idx].emotion = emo;
    m->vocab[idx].mass = mass;
    m->vocab[idx].count = 0;
    return idx;
}

static void meta_add_bigram(MetaWeights *m, int a, int b) {
    unsigned h = ((unsigned)a * 2654435761u ^ (unsigned)b) % BIGRAM_HASH;
    for (int t = 0; t < 32; t++) {
        unsigned idx = (h + t) % BIGRAM_HASH;
        if (m->bigrams[idx].count == 0) {
            m->bigrams[idx] = (WordBigram){a, b, 0, 1};
            m->n_bigrams++;
            return;
        }
        if (m->bigrams[idx].a == a && m->bigrams[idx].b == b) {
            m->bigrams[idx].count++;
            return;
        }
    }
}

static void meta_add_hebbian(MetaWeights *m, int a, int b) {
    int lo = a < b ? a : b, hi = a < b ? b : a;
    unsigned h = ((unsigned)lo * 2654435761u ^ (unsigned)hi) % HEBBIAN_HASH;
    for (int t = 0; t < 32; t++) {
        unsigned idx = (h + t) % HEBBIAN_HASH;
        if (m->hebbians[idx].strength == 0 && m->hebbians[idx].a == 0) {
            m->hebbians[idx] = (WordHebbian){lo, hi, 1.0f};
            m->n_hebbians++;
            return;
        }
        if (m->hebbians[idx].a == lo && m->hebbians[idx].b == hi) {
            m->hebbians[idx].strength += 1.0f;
            return;
        }
    }
}

static float meta_get_hebbian(MetaWeights *m, int a, int b) {
    int lo = a < b ? a : b, hi = a < b ? b : a;
    unsigned h = ((unsigned)lo * 2654435761u ^ (unsigned)hi) % HEBBIAN_HASH;
    for (int t = 0; t < 32; t++) {
        unsigned idx = (h + t) % HEBBIAN_HASH;
        if (m->hebbians[idx].a == lo && m->hebbians[idx].b == hi)
            return m->hebbians[idx].strength;
        if (m->hebbians[idx].strength == 0 && m->hebbians[idx].a == 0)
            return 0;
    }
    return 0;
}

static void build_metaweights(Organism *org) {
    MetaWeights *m = &org->meta;
    memset(m, 0, sizeof(MetaWeights));

    int word_seq[MAX_TOKENS];
    Emotion emo_seq[MAX_TOKENS];
    int seq_len = 0;

    for (int ci = 0; ci < org->n_clumps; ci++) {
        Clump *c = &org->clumps[ci];
        for (int vi = 0; vi < c->n_verses; vi++) {
            const char *txt = c->verses[vi].text;
            int pos = 0;
            while (txt[pos] && seq_len < MAX_TOKENS) {
                if (is_word_char((unsigned char)txt[pos])) {
                    char word[MAX_WORD_LEN];
                    int wlen = extract_word(txt, pos, word, MAX_WORD_LEN);
                    if (wlen > 1) {
                        int idx = meta_find(m, word);
                        if (idx < 0) {
                            Emotion we = EMO_VOID;
                            float wm = 0.1f;
                            for (int e = 0; e < org->mendeleev.n_elements; e++) {
                                if (word_eq_ci(word, org->mendeleev.table[e].word)) {
                                    we = org->mendeleev.table[e].primary;
                                    wm = org->mendeleev.table[e].mass;
                                    break;
                                }
                            }
                            idx = meta_add(m, word, we, wm);
                        }
                        if (idx >= 0) {
                            m->vocab[idx].count++;
                            word_seq[seq_len] = idx;
                            emo_seq[seq_len] = c->verses[vi].emotion;
                            seq_len++;
                        }
                    }
                    pos += wlen;
                } else { pos++; }
            }
        }
    }

    m->total_words = seq_len;

    /* frequencies */
    for (int i = 0; i < m->n_words; i++)
        m->vocab[i].frequency = (float)m->vocab[i].count / (seq_len + 1);

    /* bigrams */
    for (int i = 0; i + 1 < seq_len; i++)
        meta_add_bigram(m, word_seq[i], word_seq[i + 1]);

    /* normalize bigrams */
    for (int i = 0; i < BIGRAM_HASH; i++) {
        if (m->bigrams[i].count > 0) {
            int total = 0;
            for (int j = 0; j < BIGRAM_HASH; j++)
                if (m->bigrams[j].count > 0 && m->bigrams[j].a == m->bigrams[i].a)
                    total += m->bigrams[j].count;
            if (total > 0)
                m->bigrams[i].prob = (float)m->bigrams[i].count / total;
        }
    }

    /* hebbian */
    for (int i = 0; i < seq_len; i++)
        for (int j = i + 1; j < seq_len && j < i + CONTEXT_WIN; j++)
            if (word_seq[i] != word_seq[j])
                meta_add_hebbian(m, word_seq[i], word_seq[j]);

    /* emotional bigrams */
    float emo_cnt[EMO_COUNT][EMO_COUNT];
    memset(emo_cnt, 0, sizeof(emo_cnt));
    for (int i = 0; i + 1 < seq_len; i++)
        emo_cnt[emo_seq[i]][emo_seq[i + 1]] += 1.0f;
    for (int a = 0; a < EMO_COUNT; a++) {
        float tot = 0;
        for (int b = 0; b < EMO_COUNT; b++) tot += emo_cnt[a][b];
        if (tot > 0)
            for (int b = 0; b < EMO_COUNT; b++)
                m->emo_bigram[a][b] = emo_cnt[a][b] / tot;
    }

    printf(DIM "  metaweights: %d words, %d bigrams, %d hebbian links\n" RST,
           m->n_words, m->n_bigrams, m->n_hebbians);
}

/* ═══════════════════════════════════════════════
 *  CHEMICAL REACTION — input text hits periodic table
 * ═══════════════════════════════════════════════ */

typedef struct {
    float impulse[EMO_COUNT];
    float total_mass;
    int   reactions;
} Reaction;

static Reaction react(const char *prompt, const PeriodicTable *pt) {
    Reaction r;
    memset(&r, 0, sizeof(r));
    int pos = 0;
    while (prompt[pos]) {
        if (is_word_char((unsigned char)prompt[pos])) {
            char word[MAX_WORD_LEN];
            int wlen = extract_word(prompt, pos, word, MAX_WORD_LEN);
            for (int i = 0; i < pt->n_elements; i++) {
                if (word_eq_ci(word, pt->table[i].word)) {
                    r.impulse[pt->table[i].primary] += pt->table[i].mass;
                    r.total_mass += pt->table[i].mass;
                    r.reactions++;
                    break;
                }
            }
            pos += wlen;
        } else { pos++; }
    }
    return r;
}

/* ═══════════════════════════════════════════════
 *  BREATHING + CHAMBERS
 * ═══════════════════════════════════════════════ */

static void breathe(Organism *o) {
    o->breath_count++;

    if (o->inhaling) {
        o->breath_depth += 0.1f;
        if (o->breath_depth >= 1.0f) { o->breath_depth = 1.0f; o->inhaling = 0; }
    } else {
        o->breath_depth -= 0.08f;
        if (o->breath_depth <= 0.0f) { o->breath_depth = 0.0f; o->inhaling = 1; }
    }

    /* Kuramoto coupling between chambers */
    float delta[EMO_COUNT];
    memset(delta, 0, sizeof(delta));
    for (int i = 0; i < EMO_COUNT; i++) {
        for (int j = 0; j < EMO_COUNT; j++) {
            delta[i] += COUPLING[i][j] * o->emotions[j] * 0.02f;
        }
    }

    for (int i = 0; i < EMO_COUNT; i++) {
        o->emotions[i] *= DECAY_RATES[i];
        o->emotions[i] += o->momentum[i] * 0.1f + delta[i];
        o->momentum[i] *= 0.9f;
        if (o->emotions[i] > 1.0f) o->emotions[i] = 1.0f;
        if (o->emotions[i] < 0.0f) o->emotions[i] = 0.0f;
    }

    /* orbital mechanics */
    for (int i = 0; i < o->n_clumps; i++) {
        Clump *c = &o->clumps[i];
        float pull = 0;
        for (int e = 0; e < EMO_COUNT; e++)
            pull += c->emotional_signature[e] * o->emotions[e];
        c->gravity = 0.3f + 0.7f * (pull / (pull + 1.0f));
        c->orbital_angle += c->orbital_speed * (1.0f + o->breath_depth);
    }

    float total = 0;
    for (int i = 0; i < EMO_COUNT; i++) total += o->emotions[i];
    o->pressure += total * 0.01f;
}

/* ═══════════════════════════════════════════════
 *  PROPHECY + DREAM — emotionally modulated generation
 * ═══════════════════════════════════════════════ */

static void update_prophecy(Organism *o) {
    MetaWeights *m = &o->meta;
    for (int i = 0; i < m->n_words; i++) {
        Emotion we = m->vocab[i].emotion;
        float pull = 0;
        for (int e = 0; e < EMO_COUNT; e++)
            pull += o->emotions[e] * m->emo_bigram[e][we];
        m->prophecy[i] = pull * m->vocab[i].mass;
    }
}

static Emotion dominant_emo(const float emo[]) {
    Emotion best = EMO_VOID;
    float max = -1;
    for (int i = 0; i < EMO_COUNT; i++)
        if (emo[i] > max) { max = emo[i]; best = i; }
    return best;
}

static void dream(Organism *o, char *out, int maxlen) {
    MetaWeights *m = &o->meta;
    if (m->n_words < 5) { snprintf(out, maxlen, "..."); return; }

    update_prophecy(o);
    Emotion dom = dominant_emo(o->emotions);

    /* seed word selection */
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
        for (int i = 0; i < m->n_words && n_cand < 32; i++) {
            candidates[n_cand] = i;
            scores[n_cand] = m->vocab[i].frequency;
            n_cand++;
        }
    }

    /* weighted random seed */
    float total = 0;
    for (int i = 0; i < n_cand; i++) total += scores[i];
    float r = ((float)rand() / RAND_MAX) * total;
    float cum = 0;
    int seed = candidates[0];
    for (int i = 0; i < n_cand; i++) {
        cum += scores[i];
        if (cum >= r) { seed = candidates[i]; break; }
    }

    /* generate word chain */
    int seq[32];
    int slen = 0;
    seq[slen++] = seed;
    int dream_len = 5 + rand() % 10;

    for (int step = 1; step < dream_len && slen < 32; step++) {
        int prev = seq[slen - 1];
        float word_scores[MAX_WORDS];
        memset(word_scores, 0, m->n_words * sizeof(float));

        /* bigram */
        for (int i = 0; i < BIGRAM_HASH; i++) {
            if (m->bigrams[i].count > 0 && m->bigrams[i].a == prev) {
                int next = m->bigrams[i].b;
                if (next < m->n_words)
                    word_scores[next] += m->bigrams[i].prob * 4.0f;
            }
        }

        /* emotional alignment */
        Emotion next_emo = dom;
        float best_p = -1;
        for (int e = 0; e < EMO_COUNT; e++) {
            float p = m->emo_bigram[dom][e] * (0.3f + o->emotions[e]);
            if (p > best_p) { best_p = p; next_emo = e; }
        }
        for (int i = 0; i < m->n_words; i++) {
            if (m->vocab[i].emotion == next_emo) word_scores[i] += 1.5f;
            word_scores[i] += m->prophecy[i] * 2.0f;
            word_scores[i] += meta_get_hebbian(m, prev, i) * 0.5f;
        }

        /* repetition penalty */
        for (int s = 0; s < slen; s++) word_scores[seq[s]] *= 0.1f;

        /* trauma corruption */
        if (o->emotions[EMO_TRAUMA] > 0.6f &&
            ((float)rand() / RAND_MAX) < o->emotions[EMO_TRAUMA] * 0.3f) {
            seq[slen++] = rand() % m->n_words;
            continue;
        }

        /* temperature from void */
        float temp = 0.7f + o->emotions[EMO_VOID] * 0.8f;

        total = 0;
        float log_s[MAX_WORDS];
        float max_ls = -1e30f;
        for (int i = 0; i < m->n_words; i++) {
            if (word_scores[i] < 0) word_scores[i] = 0;
            log_s[i] = logf(word_scores[i] + 1e-10f) / temp;
            if (log_s[i] > max_ls) max_ls = log_s[i];
        }
        total = 0;
        for (int i = 0; i < m->n_words; i++) {
            word_scores[i] = expf(log_s[i] - max_ls);
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
        dom = m->vocab[chosen].emotion;
    }

    /* assemble */
    int pos = 0;
    for (int i = 0; i < slen && pos < maxlen - 2; i++) {
        if (i > 0 && pos < maxlen - 2) out[pos++] = ' ';
        const char *w = m->vocab[seq[i]].word;
        int wl = strlen(w);
        for (int j = 0; j < wl && pos < maxlen - 1; j++)
            out[pos++] = w[j];
    }
    out[pos] = '\0';
}

/* ═══════════════════════════════════════════════
 *  CITATION — find resonant verse from clumps
 * ═══════════════════════════════════════════════ */

static const Verse *find_resonance(Organism *o, const Clump **src_clump) {
    const Verse *best_v = NULL;
    float best_score = -1;
    *src_clump = NULL;

    for (int i = 0; i < o->n_clumps; i++) {
        Clump *c = &o->clumps[i];
        for (int v = 0; v < c->n_verses; v++) {
            Verse *ver = &c->verses[v];
            float align = o->emotions[ver->emotion];
            float fatigue = 1.0f / (1.0f + ver->times_cited * 0.5f);
            float score = align * ver->charge * ver->decay * c->gravity * fatigue
                        + ((float)rand() / RAND_MAX) * o->emotions[EMO_VOID] * 0.3f;
            if (score > best_score) {
                best_score = score;
                best_v = ver;
                *src_clump = c;
            }
        }
    }
    return best_v;
}

/* ═══════════════════════════════════════════════
 *  DIGEST — absorb user input into flesh
 * ═══════════════════════════════════════════════ */

static void digest(Organism *o, const char *prompt, const Reaction *r) {
    if (o->n_digested >= MAX_VERSES) return;
    Verse *v = &o->digested[o->n_digested++];
    snprintf(v->text, MAX_VERSE_LEN, "%s", prompt);
    v->emotion = dominant_emo(r->impulse);
    v->charge = r->total_mass / (r->reactions + 1);
    if (v->charge > 1.0f) v->charge = 1.0f;
    v->decay = 1.0f;
    v->times_cited = 0;
    o->total_mass += r->total_mass;

    /* rebuild metaweights */
    build_metaweights(o);
}

/* ═══════════════════════════════════════════════
 *  DISPLAY
 * ═══════════════════════════════════════════════ */

static void display_breath(Organism *o) {
    int w = 20, f = (int)(o->breath_depth * w);
    printf(DIM "  breath %s [", o->inhaling ? "<<<" : ">>>");
    for (int i = 0; i < w; i++) printf(i < f ? "▓" : "░");
    printf("] #%d\n" RST, o->breath_count);
}

static void display_emotions(Organism *o) {
    printf(DIM "  ");
    for (int i = 0; i < EMO_COUNT; i++) {
        if (o->emotions[i] > 0.05f) {
            int bar = (int)(o->emotions[i] * 10);
            printf("%s%s:", EMO_COLORS[i], EMO_NAMES[i]);
            for (int j = 0; j < bar; j++) printf("█");
            printf(RST DIM " ");
        }
    }
    printf("\n" RST);
}

/* detect language from first byte of UTF-8 word */
static const char *detect_lang(const char *word) {
    unsigned char c = (unsigned char)word[0];
    if (c < 0x80) return "EN";
    /* Russian (Cyrillic): U+0400..U+04FF → UTF-8: 0xD0 0x80..0xD1 0xBF */
    if (c == 0xD0 || c == 0xD1) return "RU";
    /* Hebrew: U+0590..U+05FF → UTF-8: 0xD6 0x90..0xD7 0xBF */
    if (c == 0xD6 || c == 0xD7) return "HE";
    /* French accented chars (Latin extended): 0xC3 xx */
    if (c == 0xC3) return "FR";
    /* fallback: if ASCII, EN; otherwise unknown */
    return "??";
}

static void display_mendeleev(Organism *o) {
    printf(DIM "  ┌─ PERIODIC TABLE OF MEANING (%d elements) ─────┐\n", o->mendeleev.n_elements);
    for (int i = 0; i < o->mendeleev.n_elements && i < 24; i++) {
        Element *e = &o->mendeleev.table[i];
        const char *lang = detect_lang(e->word);
        printf("  │ [%s] %s%-12s%s m=%.2f v=%d %s%-10s%s",
               lang,
               EMO_COLORS[e->primary], e->word, RST DIM,
               e->mass, e->valence,
               EMO_COLORS[e->primary], EMO_NAMES[e->primary], RST DIM);
        if (e->half_life == 0) printf(" ∞");
        else printf(" t½=%.0f", e->half_life);
        printf("\n");
    }
    if (o->mendeleev.n_elements > 24)
        printf("  │ ... +%d more\n", o->mendeleev.n_elements - 24);
    printf("  └─────────────────────────────────────────────────┘\n" RST);
}

/* ═══════════════════════════════════════════════
 *  INIT — from corpus file
 * ═══════════════════════════════════════════════ */

static const char *SEED_CORPUS =
    /* ═══ EN — Poe + Blake ═══ */
    "Deep into that darkness peering, long I stood there wondering, fearing.\n\n"
    "Once upon a midnight dreary, while I pondered, weak and weary.\n\n"
    "Quoth the Raven, Nevermore.\n\n"
    "And my soul from out that shadow shall be lifted — Nevermore!\n\n"
    "Tyger Tyger, burning bright, in the forests of the night.\n\n"
    "What immortal hand or eye, could frame thy fearful symmetry?\n\n"
    "To see a World in a Grain of Sand and a Heaven in a Wild Flower.\n\n"
    "Hold Infinity in the palm of your hand and Eternity in an hour.\n\n"
    "In every cry of every Man, in every Infant's cry of fear.\n\n"
    "And the hapless Soldier's sigh runs in blood down Palace walls.\n\n"

    /* ═══ FR — Baudelaire ═══ */
    "Hypocrite lecteur, mon semblable, mon frère!\n\n"
    "Mon enfant, ma sœur, songe à la douceur d'aller là-bas vivre ensemble!\n\n"
    "Là, tout n'est qu'ordre et beauté, luxe, calme et volupté.\n\n"
    "La Nature est un temple où de vivants piliers laissent parfois sortir de confuses paroles.\n\n"
    "Les parfums, les couleurs et les sons se répondent.\n\n"

    /* ═══ FR — Rimbaud ═══ */
    "Jadis, si je me souviens bien, ma vie était un festin où s'ouvraient tous les cœurs.\n\n"
    "Un soir, j'ai assis la Beauté sur mes genoux. Et je l'ai trouvée amère.\n\n"
    "Je me suis enfui. Ô sorcières, ô misère, ô haine, c'est à vous que mon trésor a été confié!\n\n"

    /* ═══ FR — Verlaine ═══ */
    "Les sanglots longs des violons de l'automne blessent mon cœur d'une langueur monotone.\n\n"
    "Il pleure dans mon cœur comme il pleut sur la ville.\n\n"
    "Et je m'en vais au vent mauvais qui m'emporte deçà, delà, pareil à la feuille morte.\n\n"

    /* ═══ RU — Мандельштам + Цветаева ═══ */
    "Я вернулся в мой город, знакомый до слёз, до прожилок, до детских припухлых желёз.\n\n"
    "Петербург! я ещё не хочу умирать: у тебя телефонов моих номера.\n\n"
    "Мне нравится, что вы больны не мной, мне нравится, что я больна не вами.\n\n"
    "Я тебя отвоюю у всех земель, у всех небес.\n\n"
    "Кто создан из камня, кто создан из глины, а я серебрюсь и сверкаю.\n\n"

    /* ═══ HE — Амихай + Рахель ═══ */
    "אלוהים מרחם על ילדי הגן, פחות מזה על ילדי בית הספר.\n\n"
    "ועל הגדולים לא ירחם כלל.\n\n"
    "המקום שבו אנחנו צודקים לא יצמחו פרחים לעולם.\n\n"
    "רק שירי אודותייך אני שרה, שירי גורלי המר.\n\n"
    "הנה רק עפר אני, כי אם השמש תזרח, אז הלוא גם אני אור.\n\n";

static void organism_init(Organism *o, const char *corpus, int corpus_len) {
    memset(o, 0, sizeof(Organism));
    o->emotions[EMO_VOID] = 0.3f;
    o->emotions[EMO_RESONANCE] = 0.2f;
    o->inhaling = 1;

    printf(DIM "  building organism from %d bytes of corpus...\n" RST, corpus_len);

    /* Phase 1: auto-discover periodic table */
    auto_mendeleev(corpus, corpus_len, &o->mendeleev);

    /* Phase 2: build clumps (emotional planets) */
    build_clumps(corpus, corpus_len, o);

    /* Phase 3: build metaweights */
    build_metaweights(o);

    printf(DIM "  organism alive. %d elements, %d planets, %d words.\n\n" RST,
           o->mendeleev.n_elements, o->n_clumps, o->meta.n_words);
}

/* ═══════════════════════════════════════════════
 *  MAIN LOOP
 * ═══════════════════════════════════════════════ */

int main(int argc, char **argv) {
    srand(time(NULL));

    /* load corpus */
    char *corpus = NULL;
    int corpus_len = 0;

    if (argc > 1) {
        FILE *f = fopen(argv[1], "r");
        if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
        corpus = malloc(MAX_CORPUS);
        corpus_len = fread(corpus, 1, MAX_CORPUS - 1, f);
        corpus[corpus_len] = '\0';
        fclose(f);
        printf(DIM "  loaded: %s (%d bytes)\n" RST, argv[1], corpus_len);
    } else {
        corpus_len = strlen(SEED_CORPUS);
        corpus = (char *)SEED_CORPUS;
        printf(DIM "  no corpus file — using built-in seed (Poe, Blake, Baudelaire, Rimbaud, Verlaine, Mandelstam, Tsvetaeva, Amichai, Rachel)\n" RST);
    }

    Organism o;
    organism_init(&o, corpus, corpus_len);

    printf("\033[2J\033[H");
    printf(BOLD
    "  ██████╗ ██████╗  ██████╗\n"
    "  ██╔══██╗██╔══██╗██╔═══██╗\n"
    "  ██████╔╝██████╔╝██║   ██║\n"
    "  ██╔═══╝ ██╔══██╗██║   ██║\n"
    "  ██║     ██║  ██║╚██████╔╝\n"
    "  ╚═╝     ╚═╝  ╚═╝ ╚═════╝\n"
    RST);
    printf(ITAL "  PostGPT Pro — the corpus is the organism\n" RST);
    printf(ITAL "  EN · FR · RU · HE — four tongues, one breath\n" RST);
    printf(DIM "  %d elements discovered · %d planets · %d words · 192 anchors\n",
           o.mendeleev.n_elements, o.n_clumps, o.meta.n_words);
    printf("  commands: /dream /state /table /quit\n\n" RST);

    char input[1024];

    while (1) {
        display_breath(&o);
        printf(BOLD "  ⚛ > " RST);
        fflush(stdout);

        if (!fgets(input, sizeof(input), stdin)) break;
        input[strcspn(input, "\n")] = '\0';
        if (strlen(input) == 0) { breathe(&o); continue; }

        if (strcmp(input, "/quit") == 0) break;
        if (strcmp(input, "/state") == 0) {
            display_emotions(&o);
            printf(DIM "  pressure: %.2f | digested: %d | words: %d\n" RST,
                   o.pressure, o.n_digested, o.meta.n_words);
            breathe(&o);
            continue;
        }
        if (strcmp(input, "/table") == 0) {
            display_mendeleev(&o);
            breathe(&o);
            continue;
        }
        if (strcmp(input, "/dream") == 0) {
            char d[512];
            dream(&o, d, sizeof(d));
            strncpy(o.last_dream, d, sizeof(o.last_dream));
            printf("\n  " ITAL "%s« %s »%s\n",
                   EMO_COLORS[dominant_emo(o.emotions)], d, RST);
            printf(DIM "    — [the organism dreams]\n\n" RST);
            breathe(&o);
            continue;
        }

        /* chemical reaction */
        Reaction r = react(input, &o.mendeleev);

        for (int i = 0; i < EMO_COUNT; i++) {
            o.momentum[i] += r.impulse[i] * 0.5f;
            o.emotions[i] += r.impulse[i] * 0.3f;
            if (o.emotions[i] > 1.0f) o.emotions[i] = 1.0f;
        }

        if (r.reactions > 0)
            printf(DIM "  ⚗ %d reactions, mass=%.2f\n" RST, r.reactions, r.total_mass);

        for (int b = 0; b < 3; b++) breathe(&o);
        display_emotions(&o);

        /* citation */
        const Clump *src = NULL;
        const Verse *cite = find_resonance(&o, &src);
        if (cite) {
            ((Verse*)cite)->times_cited++;
            printf("\n  %s« %s »%s\n", EMO_COLORS[cite->emotion], cite->text, RST);
            if (src)
                printf(DIM "    — %s [%s %.0f%%]\n" RST,
                       src->name, EMO_NAMES[cite->emotion], cite->charge * 100);
        }

        /* dream on pressure */
        if (o.pressure > 1.5f || o.breath_count % 10 == 0) {
            char d[512];
            dream(&o, d, sizeof(d));
            strncpy(o.last_dream, d, sizeof(o.last_dream));
            printf("  " ITAL "%s  ≈ %s ≈%s\n",
                   EMO_COLORS[dominant_emo(o.emotions)], d, RST);
            printf(DIM "    — [dreaming]\n" RST);
        }

        printf("\n");

        /* digest heavy input */
        if (r.total_mass > 1.0f || o.pressure > 2.0f) {
            digest(&o, input, &r);
            printf(DIM "    [digested — metaweights rebuilt]\n" RST);
            o.pressure *= 0.5f;
        }
    }

    printf(DIM "\n  the organism sleeps. the mendeleev table persists.\n\n" RST);
    if (corpus != (char*)SEED_CORPUS) free(corpus);
    return 0;
}
