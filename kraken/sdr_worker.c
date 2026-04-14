/*
* SDR acquisition I/O worker
*
* 1) opening the SDR device
* 2) reading sample from the SDR
* 3) packing the raw samples as a frame unit, and returning it to stdout
*/ 

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h> // fixed length integer dtypes - ensuring the strict byte syze
#include <string.h> 
#include <signal.h> // ctrl+C 상황에서 worker를 죽이지 않고, 각 thread가 loop을 종료하도록 termination flag 전달
#include <unistd.h> // Unix/POSIX system call/utils
#include <pthread.h> // threading
#include "rtl-sdr.h" // SDR API

#define MAX_DEVICES 5 // Number of maximum channels in KrakenSDR
#define MAGIC 0x534452 // File signature - 'SDR' (SDR에 의해서 만들어 진 것이라는)

// FrameHeader: C -> Python, stack 임시객체
typedef struct{
    uint32_t magic; // checking the frame initiation
    uint16_t channel_id;
    uint16_t flags;
    uint64_t frame_id;
    uint32_t sample_bytes; // byte length of following samples, 2 * N_daq (I/Q sampling)
} __attribute__((packed)) FrameHeader;

// DeviceCotext: handing the SDR device
// located in the main stack, thread refers the memory adress
typedef struct{
    int device_idx;
    uint32_t sample_rate;
    uint32_t center_freq;
    int gain_db;
    uint32_t n_daq; // number of complex IQ samples in single block
    uint64_t frame_id;
    rtlsdr_dev_t *dev; // rtlsdr object
    pthread_t thread;
} DeviceContext;

// g_running: 1 - keep running / 0 - escaped from loop
// *static: storing the values at a static storage (프로그램 시작 시 생성, 종료 시 파괴)
// *volatile: compiler가 register에 cahcning해 둔 후, 다시 값을 읽지 않는 것을 방지 (게속 체크?)
static volatile sig_atomic_t g_running = 1;
// stdout: shared resource, protected by MUTEX (prevent tangling between channels) 
static pthread_mutex_t g_write_mutex = PTHREAD_MUTEX_INITIALIZER; 

// on_sigint: signal handler -> change global flag
static void on_sigint(int signum){
    (void) signum;
    g_running = 0; 
}

static int config_device(DeviceContext *ctx){
    // *ctx: struct pointer of DeviceContext
    // ctx.: return attributes in struct itself
    // ctx->: return attributes directed by pointer, equivalent to (*ctx).
    int r = 0;

    //&ctx->dev: put opened device handle 'dev' to ctx->dev
    r = rtlsdr_open(&ctx->dev, (uint32_t)ctx->device_idx);
    if (r < 0) {
        fprintf(stderr, "Failed to open device %d\n", ctx->device_idx);
        return -1;
    }
    // sample rate
    r = rtlsdr_set_sample_rate(ctx->dev, ctx->sample_rate);
    if (r < 0) {
        fprintf(stderr, "Failed to set sample rate on device %d\n", ctx->device_idx);
        return -1;
    }
    // center frequency
    r = rtlsdr_set_center_freq(ctx->dev, ctx->center_freq);
    if (r < 0) {
        fprintf(stderr, "Failed to set center freq on device %d\n", ctx->device_idx);
        return -1;
    }
    // manual gain mode
    r = rtlsdr_set_tuner_gain_mode(ctx->dev, 1);
    if (r < 0) {
        fprintf(stderr, "Failed to enable manual gain on device %d\n", ctx->device_idx);
        return -1;
    }
    // set ADC gain [dB] 
    r = rtlsdr_set_tuner_gain(ctx->dev, ctx->gain_db * 10); // libsdr uses tenths of dB
    if (r < 0) {
        fprintf(stderr, "Failed to set gain on device %d\n", ctx->device_idx);
        return -1;
    }
    // reset device buffer
    r = rtlsdr_reset_buffer(ctx->dev);
    if (r < 0) {
        fprintf(stderr, "Failed to reset buffer on device %d\n", ctx->device_idx);
        return -1;
    }

    return 0;
}

/* release device handle */
// 내가 malloc 한 것이 아니라, rtlsdr_open에서 한 객체임.
static void close_device(DeviceContext *ctx) {
    if (ctx->dev) {
        rtlsdr_close(ctx->dev);
        ctx->dev = NULL;
    }
}

/* read samples from SDR */
static void *reader_thread(void *arg) {
    DeviceContext *ctx = (DeviceContext *)arg;

    const uint32_t sample_bytes = ctx->n_daq * 2; // I,Q each 1 byte
    uint8_t *buf = (uint8_t *)malloc(sample_bytes); // assign buffer (allocate a memory only once in each thread)
    if (!buf) {
        fprintf(stderr, "malloc failed for device %d\n", ctx->device_idx);
        return NULL;
    }

    while (g_running) {
        int n_read = 0;
        int r = rtlsdr_read_sync(ctx->dev, buf, (int)sample_bytes, &n_read);
        // temporary, we use "rtlsdr_read_sync" (simple structure, memory handling)
        if (r < 0) {
            fprintf(stderr, "read_sync failed on device %d\n", ctx->device_idx);
            break;
        }
        if ((uint32_t)n_read != sample_bytes) {
            fprintf(stderr, "[read] device %d, iteration = %d (%u bytes) \n",
                    n_read, sample_bytes, ctx->device_idx);
            continue;
        }

        FrameHeader hdr;
        hdr.magic = MAGIC;
        hdr.channel_id = (uint16_t)ctx->device_idx;
        hdr.flags = 0;
        hdr.frame_id = ctx->frame_id++;
        hdr.sample_bytes = sample_bytes;

        // return header+samples to stdout
        pthread_mutex_lock(&g_write_mutex); // stdout is a shared resource, then memory lock should be preceded.
        // fwrite(*original_buffer, byte_size, count, *file), return: file에 쓰기 성공한 데이터 개수
        // fwrite: store in C library stdout buffer
        size_t w1 = fwrite(&hdr, sizeof(hdr), 1, stdout); // sizeof(hdr): byte size of hdr
        size_t w2 = fwrite(buf, sample_bytes, 1, stdout);
        fflush(stdout); // fflush: export to the real file discripter (stdout)
        pthread_mutex_unlock(&g_write_mutex);

        if (w1 != sizeof(hdr) || w2 != sample_bytes) {
            fprintf(stderr, "stdout write failed on device %d\n", ctx->device_idx);
            break;
        }
    }

    free(buf); // release memory
    return NULL;
}

static int parse_device_list(const char *s, int *out, int max_n) {
    int n = 0;
    char *tmp = strdup(s); // heap에 copy 생성
    char *tok = strtok(tmp, ",");// in-place separation "0,1,2" -> {0,1,2}
    while (tok && n < max_n) {
        out[n++] = atoi(tok); // convert to number
        tok = strtok(NULL, ",");
    }
    free(tmp);
    return n;
}

int main(int argc, char **argv) {
    signal(SIGINT, on_sigint);

    // default params
    char device_arg[128] = "0,1";
    uint32_t sample_rate = 2560000;
    uint32_t center_freq = 1420000000;
    int gain_db = 50;
    uint32_t n_daq = 131072; // 2^17

    // argv parsing
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--devices") && i + 1 < argc) {
            strncpy(device_arg, argv[++i], sizeof(device_arg) - 1);
        } else if (!strcmp(argv[i], "--sample-rate") && i + 1 < argc) {
            sample_rate = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--center-freq") && i + 1 < argc) {
            center_freq = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--gain") && i + 1 < argc) {
            gain_db = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--n_daq") && i + 1 < argc) {
            n_daq = (uint32_t)strtoul(argv[++i], NULL, 10);
        }
    }

    int device_ids[MAX_DEVICES];
    int n_dev = parse_device_list(device_arg, device_ids, MAX_DEVICES);
    if (n_dev <= 0) {
        fprintf(stderr, "No devices specified\n");
        return 1;
    }

    DeviceContext ctxs[MAX_DEVICES];
    memset(ctxs, 0, sizeof(ctxs));

    for (int i = 0; i < n_dev; i++) {
        ctxs[i].device_idx = device_ids[i];
        ctxs[i].sample_rate = sample_rate;
        ctxs[i].center_freq = center_freq;
        ctxs[i].gain_db = gain_db;
        ctxs[i].n_daq = n_daq;
        ctxs[i].frame_id = 0;

        if (config_device(&ctxs[i]) != 0) {
            fprintf(stderr, "Configuration failed for device %d\n", ctxs[i].device_idx);
            return 1;
        }
    }

    for (int i = 0; i < n_dev; i++) {
        pthread_create(&ctxs[i].thread, NULL, reader_thread, &ctxs[i]);
    }

    for (int i = 0; i < n_dev; i++) {
        pthread_join(ctxs[i].thread, NULL);
    }

    for (int i = 0; i < n_dev; i++) {
        close_device(&ctxs[i]);
    }

    return 0;
}
