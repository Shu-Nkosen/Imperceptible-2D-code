// R8 sweep presenter (GLFW)
// - Slate: black -> red (for sync detection), preferably slate_*.png with focus grid
// - Then 60 blocks, each block_sec long
// - In each block, alternate normal/inv textures every frame (or every repeat frame)
//
// Build example (PowerShell, run inside R8/make_movie):
//   gcc present_session.c -I"..\\..\\vcpkg\\installed\\x64-mingw-dynamic\\include" -L"..\\..\\vcpkg\\installed\\x64-mingw-dynamic\\lib" -lglfw3dll -lwinmm -lopengl32 -lgdi32 -luser32 -o present_session.exe

#include <GLFW/glfw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

typedef struct {
    int intensity;         // e.g. 4/8/12
    const char* channel;   // "R","G","B","max","min"
    const char* token;     // file suffix token: R/G/B/X/I
    const char* base_name; // e.g. rice
} Condition;

// NOTE: keep in sync with available base images in R8/make_movie (*.png)
static const char* BASE_IMAGES[] = { "rice", "nagaoka_fireworks", "hocho", "ex" };
static const int BASE_IMAGE_COUNT = 4;
static const int INTENSITIES[] = { 4, 8, 12 };
static const int INTENSITY_COUNT = 3;
static const char* CHANNELS[] = { "R", "G", "B", "min", "max" };
static const char* TOKENS[]   = { "R", "G", "B", "I",   "X"   };
static const int CHANNEL_COUNT = 5;

static void usage(void) {
    printf("Usage: present_session.exe --rate <Hz> --exp <250|125|60> --interval <n> --out-manifest <path.json> [--block-sec 6] [--slate-sec 0.5] [--padding-sec 5] [--qr-sec 3] [--repeat 1]\\n");
    printf("Note: set OS display refresh to target rate before running (esp. 120Hz).\\n");
    printf("GT QR (HP_QR.png) is shown before cond0, before cond30, and after cond59.\\n");
}

static int parse_int_arg(int argc, char** argv, const char* key, int default_value) {
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], key) == 0) return atoi(argv[i + 1]);
    }
    return default_value;
}

static double parse_double_arg(int argc, char** argv, const char* key, double default_value) {
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], key) == 0) return atof(argv[i + 1]);
    }
    return default_value;
}

static const char* parse_str_arg(int argc, char** argv, const char* key, const char* default_value) {
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], key) == 0) return argv[i + 1];
    }
    return default_value;
}

static int has_flag(int argc, char** argv, const char* key) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], key) == 0) return 1;
    }
    return 0;
}

static GLuint load_texture(const char* filename) {
    int width, height, channels;
    unsigned char* data = stbi_load(filename, &width, &height, &channels, 0);
    if (!data) {
        printf("[ERROR] stbi_load failed: %s\\n", filename);
        return 0;
    }

    GLuint texture_id;
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    GLenum format = (channels == 4) ? GL_RGBA : GL_RGB;
    glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
    stbi_image_free(data);
    return texture_id;
}

static void draw_texture(GLuint texture_id) {
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 1); glVertex2f(-1, -1);
    glTexCoord2f(1, 1); glVertex2f( 1, -1);
    glTexCoord2f(1, 0); glVertex2f( 1,  1);
    glTexCoord2f(0, 0); glVertex2f(-1,  1);
    glEnd();
    glDisable(GL_TEXTURE_2D);
}

static void draw_solid(float r, float g, float b) {
    glDisable(GL_TEXTURE_2D);
    glClearColor(r, g, b, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
}

static void draw_black_frames(GLFWwindow* window, int frames) {
    for (int f = 0; f < frames && !glfwWindowShouldClose(window); f++) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, 1);
        draw_solid(0.0f, 0.0f, 0.0f);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}

static void draw_solid_frames(GLFWwindow* window, int frames, float r, float g, float b) {
    for (int f = 0; f < frames && !glfwWindowShouldClose(window); f++) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, 1);
        draw_solid(r, g, b);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}

static int calc_frames(double sec, int refresh_hz, int interval) {
    if (sec <= 0) return 0;
    double fps = ((double)refresh_hz) / (double)interval;
    int frames = (int)(sec * fps + 0.5);
    if (frames < 1) frames = 1;
    return frames;
}

static int file_exists(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

static const char* resolve_gt_qr_path(void) {
    // Prefer precomposed full-frame asset (same geometry as embedded QR).
    static const char* full_path = "gt_qr_display.png";
    static const char* mask_path = "HP_QR.png";
    if (file_exists(full_path)) return full_path;
    if (file_exists(mask_path)) return mask_path;
    return NULL;
}

// Same placement as gen_assets.py: QR fills height x height square, centered on 1920x1080.
static void nn_resize_gray(
    const unsigned char* src, int sw, int sh,
    unsigned char* dst, int dw, int dh
) {
    for (int y = 0; y < dh; y++) {
        int sy = (int)((long long)y * sh / dh);
        if (sy >= sh) sy = sh - 1;
        for (int x = 0; x < dw; x++) {
            int sx = (int)((long long)x * sw / dw);
            if (sx >= sw) sx = sw - 1;
            dst[y * dw + x] = src[sy * sw + sx];
        }
    }
}

static GLuint upload_rgb_texture(const unsigned char* rgb, int width, int height) {
    GLuint texture_id;
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb);
    return texture_id;
}

static GLuint load_gt_qr_texture(const char* filename, int canvas_w, int canvas_h) {
    int width = 0, height = 0, channels = 0;
    unsigned char* data = stbi_load(filename, &width, &height, &channels, 0);
    if (!data) {
        printf("[ERROR] stbi_load failed: %s\\n", filename);
        return 0;
    }

    // Already a full-frame display asset: upload as-is (convert gray/RGBA -> RGB).
    if (width == canvas_w && height == canvas_h) {
        unsigned char* rgb = (unsigned char*)malloc((size_t)canvas_w * (size_t)canvas_h * 3u);
        if (!rgb) {
            stbi_image_free(data);
            return 0;
        }
        for (int i = 0; i < canvas_w * canvas_h; i++) {
            unsigned char v;
            if (channels == 1) {
                v = data[i];
            } else if (channels >= 3) {
                v = data[i * channels];
            } else {
                v = 255;
            }
            rgb[i * 3 + 0] = v;
            rgb[i * 3 + 1] = v;
            rgb[i * 3 + 2] = v;
        }
        stbi_image_free(data);
        GLuint tex = upload_rgb_texture(rgb, canvas_w, canvas_h);
        free(rgb);
        printf("[INFO] GT QR texture loaded full-frame: %s (%dx%d)\\n", filename, canvas_w, canvas_h);
        return tex;
    }

    // Mask (e.g. 330x330): compose white 1920x1080 with centered square QR (=height).
    int square = canvas_h;
    int x0 = (canvas_w - square) / 2;
    if (x0 < 0) x0 = 0;

    unsigned char* gray_src = (unsigned char*)malloc((size_t)width * (size_t)height);
    unsigned char* gray_sq = (unsigned char*)malloc((size_t)square * (size_t)square);
    unsigned char* rgb = (unsigned char*)malloc((size_t)canvas_w * (size_t)canvas_h * 3u);
    if (!gray_src || !gray_sq || !rgb) {
        free(gray_src);
        free(gray_sq);
        free(rgb);
        stbi_image_free(data);
        return 0;
    }

    for (int i = 0; i < width * height; i++) {
        if (channels == 1) {
            gray_src[i] = data[i];
        } else if (channels >= 3) {
            gray_src[i] = data[i * channels];
        } else {
            gray_src[i] = 255;
        }
    }
    stbi_image_free(data);

    nn_resize_gray(gray_src, width, height, gray_sq, square, square);
    free(gray_src);

    // White background
    memset(rgb, 255, (size_t)canvas_w * (size_t)canvas_h * 3u);
    for (int y = 0; y < square; y++) {
        for (int x = 0; x < square; x++) {
            unsigned char v = gray_sq[y * square + x];
            // Binary like gen_assets / create_white_qr
            v = (v < 128) ? 0 : 255;
            int dx = x0 + x;
            if (dx < 0 || dx >= canvas_w) continue;
            int di = (y * canvas_w + dx) * 3;
            rgb[di + 0] = v;
            rgb[di + 1] = v;
            rgb[di + 2] = v;
        }
    }
    free(gray_sq);

    GLuint tex = upload_rgb_texture(rgb, canvas_w, canvas_h);
    free(rgb);
    printf(
        "[INFO] GT QR composed from mask %s (%dx%d) -> canvas %dx%d center_square=%d x0=%d\\n",
        filename, width, height, canvas_w, canvas_h, square, x0
    );
    return tex;
}

static void draw_texture_frames(GLFWwindow* window, GLuint texture_id, int frames) {
    for (int f = 0; f < frames && !glfwWindowShouldClose(window); f++) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, 1);
        draw_texture(texture_id);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}

static GLuint try_load_texture(const char* filename) {
    if (!file_exists(filename)) return 0;
    return load_texture(filename);
}

static void draw_slate_frames(
    GLFWwindow* window,
    GLuint texture_id,
    int frames,
    float fallback_r,
    float fallback_g,
    float fallback_b
) {
    if (texture_id != 0) {
        draw_texture_frames(window, texture_id, frames);
        return;
    }
    draw_solid_frames(window, frames, fallback_r, fallback_g, fallback_b);
}

static void write_manifest(
    const char* out_path,
    int rate_hz,
    int exp,
    int interval,
    int refresh_hz,
    double block_sec,
    double slate_sec,
    double padding_sec,
    double qr_sec,
    int repeat,
    int cond_count,
    Condition* conds,
    const char* gt_qr_image
) {
    if (!out_path || strlen(out_path) == 0) return;

    {
        char tmp[MAX_PATH];
        strncpy(tmp, out_path, sizeof(tmp) - 1);
        tmp[sizeof(tmp) - 1] = '\0';
        for (char* p = tmp; *p; p++) {
            if (*p == '/' || *p == '\\') {
                char saved = *p;
                *p = '\0';
                if (strlen(tmp) > 0) {
                    CreateDirectoryA(tmp, NULL);
                }
                *p = saved;
            }
        }
    }

    FILE* f = fopen(out_path, "wb");
    if (!f) {
        printf("[WARN] cannot write manifest: %s\\n", out_path);
        return;
    }

    // QR insert points: before condition 0, 30, and after last (60)
    const int qr_before[] = { 0, 30, 60 };
    const char* qr_names[] = { "start", "mid", "end" };
    const int qr_count = 3;

    fprintf(f, "{\n");
    fprintf(f, "  \"rate_hz\": %d,\n", rate_hz);
    fprintf(f, "  \"exp\": %d,\n", exp);
    fprintf(f, "  \"interval\": %d,\n", interval);
    fprintf(f, "  \"monitor_hz\": %d,\n", refresh_hz);
    fprintf(f, "  \"block_sec\": %.6f,\n", block_sec);
    fprintf(f, "  \"slate_sec\": %.6f,\n", slate_sec);
    fprintf(f, "  \"padding_sec\": %.6f,\n", padding_sec);
    fprintf(f, "  \"gt_qr_sec\": %.6f,\n", qr_sec);
    fprintf(f, "  \"gt_qr_image\": \"%s\",\n", gt_qr_image ? gt_qr_image : "HP_QR.png");
    fprintf(f, "  \"repeat\": %d,\n", repeat);

    // start_sec_from_sync: sync = red onset; then remaining red slate + post padding + content
    double t = slate_sec + padding_sec;
    double slot_starts[3] = {0};
    int slot_i = 0;
    for (int i = 0; i <= cond_count; i++) {
        if (slot_i < qr_count && qr_before[slot_i] == i) {
            slot_starts[slot_i] = t;
            t += qr_sec;
            slot_i++;
        }
        if (i < cond_count) {
            t += block_sec;
        }
    }
    fprintf(f, "  \"gt_qr_slots\": [\n");
    for (int i = 0; i < qr_count; i++) {
        fprintf(
            f,
            "    {\"name\": \"%s\", \"insert_before_cond\": %d, \"start_sec_from_sync\": %.6f, \"duration_sec\": %.6f}%s\n",
            qr_names[i],
            qr_before[i],
            slot_starts[i],
            qr_sec,
            (i + 1 == qr_count) ? "" : ","
        );
    }
    fprintf(f, "  ],\n");

    fprintf(f, "  \"conditions\": [\n");
    for (int i = 0; i < cond_count; i++) {
        Condition* c = &conds[i];
        fprintf(
            f,
            "    {\"idx\": %d, \"image\": \"%s\", \"channel\": \"%s\", \"token\": \"%s\", \"intensity\": %d}%s\n",
            i, c->base_name, c->channel, c->token, c->intensity, (i + 1 == cond_count) ? "" : ","
        );
    }
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");
    fclose(f);
    printf("[OK] wrote manifest: %s\\n", out_path);
}

static void build_conditions(Condition* out, int* out_count) {
    int idx = 0;
    // Order: channel (outer) -> image (middle) -> intensity (inner)
    for (int ci = 0; ci < CHANNEL_COUNT; ci++) {
        for (int bi = 0; bi < BASE_IMAGE_COUNT; bi++) {
            for (int ii = 0; ii < INTENSITY_COUNT; ii++) {
                out[idx].base_name = BASE_IMAGES[bi];
                out[idx].channel = CHANNELS[ci];
                out[idx].token = TOKENS[ci];
                out[idx].intensity = INTENSITIES[ii];
                idx++;
            }
        }
    }
    *out_count = idx;
}

static void filename_for(char* out, size_t out_size, const Condition* c, const char* mode) {
    // mode: "normal" | "inv"
    snprintf(out, out_size, "%s_%d_%s%s.png", c->base_name, c->intensity, mode, c->token);
}

int main(int argc, char** argv) {
    if (has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h")) {
        usage();
        return 0;
    }

    int rate_hz = parse_int_arg(argc, argv, "--rate", 0);
    int exp = parse_int_arg(argc, argv, "--exp", 0);
    int interval = parse_int_arg(argc, argv, "--interval", 1);
    int repeat = parse_int_arg(argc, argv, "--repeat", 1);
    double block_sec = parse_double_arg(argc, argv, "--block-sec", 6.0);
    double slate_sec = parse_double_arg(argc, argv, "--slate-sec", 0.5);
    double padding_sec = parse_double_arg(argc, argv, "--padding-sec", 5.0);
    double qr_sec = parse_double_arg(argc, argv, "--qr-sec", 3.0);
    const char* out_manifest = parse_str_arg(argc, argv, "--out-manifest", "");

    if (interval < 1) interval = 1;
    if (repeat < 1) repeat = 1;
    if (qr_sec < 0.5) qr_sec = 0.5;

    timeBeginPeriod(1);

    if (!glfwInit()) {
        fprintf(stderr, "GLFW initialization failed\\n");
        timeEndPeriod(1);
        return 1;
    }

    GLFWmonitor* primary = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(primary);
    int refresh_hz = 60;
    int width = 1920;
    int height = 1080;
    if (mode) {
        refresh_hz = mode->refreshRate;
        width = mode->width;
        height = mode->height;
    }

    glfwWindowHint(GLFW_RED_BITS, mode ? mode->redBits : 8);
    glfwWindowHint(GLFW_GREEN_BITS, mode ? mode->greenBits : 8);
    glfwWindowHint(GLFW_BLUE_BITS, mode ? mode->blueBits : 8);
    glfwWindowHint(GLFW_REFRESH_RATE, refresh_hz);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(width, height, "R8 Presenter", primary, NULL);
    if (!window) {
        fprintf(stderr, "Failed to create window\\n");
        glfwTerminate();
        timeEndPeriod(1);
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(interval);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);

    printf("[INFO] monitor_hz=%d interval=%d => fps~%.3f\\n", refresh_hz, interval, ((double)refresh_hz)/(double)interval);
    if (rate_hz > 0) printf("[INFO] rate_hz(arg)=%d (OS display should match)\\n", rate_hz);
    if (exp > 0) printf("[INFO] exp(arg)=%d\\n", exp);

    Condition conds[128];
    int cond_count = 0;
    build_conditions(conds, &cond_count);
    if (cond_count != 60) {
        printf("[WARN] expected 60 conditions, got %d\\n", cond_count);
    }

    const char* gt_qr_path = resolve_gt_qr_path();
    if (!gt_qr_path) {
        printf("[ERROR] missing GT QR image. Place HP_QR.png in R8/make_movie.\\n");
        glfwDestroyWindow(window);
        glfwTerminate();
        timeEndPeriod(1);
        return 1;
    }
    printf("[INFO] GT QR image: %s\\n", gt_qr_path);

    write_manifest(
        out_manifest, rate_hz, exp, interval, refresh_hz,
        block_sec, slate_sec, padding_sec, qr_sec, repeat, cond_count, conds, gt_qr_path
    );

    const int padding_frames = calc_frames(padding_sec, refresh_hz, interval);
    const int slate_frames = calc_frames(slate_sec, refresh_hz, interval);
    const int block_frames = calc_frames(block_sec, refresh_hz, interval);
    const int qr_frames = calc_frames(qr_sec, refresh_hz, interval);

    // Texture cache: load all textures upfront (simple, avoids jitter)
    GLuint normal_tex[128];
    GLuint inv_tex[128];
    memset(normal_tex, 0, sizeof(normal_tex));
    memset(inv_tex, 0, sizeof(inv_tex));

    char fname[256];
    for (int i = 0; i < cond_count; i++) {
        filename_for(fname, sizeof(fname), &conds[i], "normal");
        normal_tex[i] = load_texture(fname);
        filename_for(fname, sizeof(fname), &conds[i], "inv");
        inv_tex[i] = load_texture(fname);
        if (normal_tex[i] == 0 || inv_tex[i] == 0) {
            printf("[ERROR] missing texture(s) for idx=%d. Did you run gen_assets.py?\\n", i);
            printf("        normal=%u inv=%u\\n", normal_tex[i], inv_tex[i]);
            glfwDestroyWindow(window);
            glfwTerminate();
            timeEndPeriod(1);
            return 1;
        }
    }

    // Compose GT QR to match embedded QR geometry (center square = height on 1920x1080).
    // Do NOT stretch HP_QR.png (e.g. 330x330) to full 16:9 — that breaks scale for pixel_acc.
    GLuint gt_qr_tex = load_gt_qr_texture(gt_qr_path, width, height);
    if (gt_qr_tex == 0) {
        printf("[ERROR] failed to load GT QR texture: %s\\n", gt_qr_path);
        glfwDestroyWindow(window);
        glfwTerminate();
        timeEndPeriod(1);
        return 1;
    }

    // Sync slate textures (focus grid). Missing files -> solid fallback.
    GLuint slate_black_tex = try_load_texture("slate_black.png");
    GLuint slate_red_tex = try_load_texture("slate_red.png");
    const int focus_grid_on = (slate_black_tex != 0 && slate_red_tex != 0) ? 1 : 0;
    if (!focus_grid_on) {
        printf(
            "[WARN] slate_black.png / slate_red.png missing or incomplete; "
            "using solid black/red (run gen_assets.py for focus grid)\\n"
        );
    }

    printf(
        "[INFO] timeline: pre_padding=%d -> slate_black=%d -> slate_red=%d -> post_padding=%d "
        "-> QR(%d)x3 interleaved with conditions focus_grid=%s\\n",
        padding_frames, slate_frames, slate_frames, padding_frames, qr_frames,
        focus_grid_on ? "on" : "off"
    );
    printf("[INFO] gt_qr_sec=%.3f (before cond0, before cond30, after last)\\n", qr_sec);

    draw_slate_frames(window, slate_black_tex, padding_frames, 0.0f, 0.0f, 0.0f);

    // Slate: black -> red (sync signal)
    draw_slate_frames(window, slate_black_tex, slate_frames, 0.0f, 0.0f, 0.0f);
    draw_slate_frames(window, slate_red_tex, slate_frames, 1.0f, 0.0f, 0.0f);

    draw_slate_frames(window, slate_black_tex, padding_frames, 0.0f, 0.0f, 0.0f);

    printf("[INFO] start conditions: %d blocks, block_frames=%d, repeat=%d\\n", cond_count, block_frames, repeat);

    // QR before first condition
    printf("[INFO] GT QR slot: start\\n");
    draw_texture_frames(window, gt_qr_tex, qr_frames);

    for (int i = 0; i < cond_count && !glfwWindowShouldClose(window); i++) {
        if (i == 30) {
            printf("[INFO] GT QR slot: mid\\n");
            draw_texture_frames(window, gt_qr_tex, qr_frames);
        }

        GLuint t_normal = normal_tex[i];
        GLuint t_inv = inv_tex[i];
        int flip = 0;
        for (int f = 0; f < block_frames && !glfwWindowShouldClose(window); f++) {
            if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, 1);
            if (repeat == 1) {
                flip = f & 1;
            } else {
                flip = ((f / repeat) & 1);
            }
            draw_texture(flip ? t_inv : t_normal);
            glfwSwapBuffers(window);
            glfwPollEvents();
        }
        if ((i + 1) % 10 == 0 || (i + 1) == cond_count) {
            printf("[INFO] progress: %d/%d\\n", i + 1, cond_count);
        }
    }

    // QR after last condition
    if (!glfwWindowShouldClose(window)) {
        printf("[INFO] GT QR slot: end\\n");
        draw_texture_frames(window, gt_qr_tex, qr_frames);
    }

    if (gt_qr_tex) glDeleteTextures(1, &gt_qr_tex);
    for (int i = 0; i < cond_count; i++) {
        if (normal_tex[i]) glDeleteTextures(1, &normal_tex[i]);
        if (inv_tex[i]) glDeleteTextures(1, &inv_tex[i]);
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    timeEndPeriod(1);
    printf("[OK] finished\\n");
    return 0;
}

