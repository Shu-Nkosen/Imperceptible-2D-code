// 実行方法
// gcc make_movie_4photo.c -DCLIP_MARGIN=14 \
//   -I"c:\Users\visulab\shu_kondo\Imperceptible-2D-code\vcpkg\installed\x64-mingw-dynamic\include" \
//   -L"c:\Users\visulab\shu_kondo\Imperceptible-2D-code\vcpkg\installed\x64-mingw-dynamic\lib" \
//   -lglfw3dll -lwinmm -lopengl32 -lgdi32 -luser32 -o make_movie_4photo.exe

#include <GLFW/glfw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <time.h>
#include <direct.h> 
#ifndef SELECTED_IMAGE
#define SELECTED_IMAGE 4
#endif

#ifndef BRIGHTNESS_DECREASE
#define BRIGHTNESS_DECREASE 2
#endif

#ifndef INTERVAL
#define INTERVAL 1
#endif

#ifndef CLIP_MARGIN
#define CLIP_MARGIN 6
#endif

// COLOR はトークンで持つ（X, I, R, G, B など）
#ifndef COLOR
#define COLOR X
#endif

// トークン→文字列
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)


#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"  // 画像読み込み用（https://github.com/nothings/stb）


void print_current_directory(void) {
    char cwd[MAX_PATH];
    if (_getcwd(cwd, sizeof(cwd))) {
        printf("[DEBUG] Current working directory: %s\n", cwd);
    } else {
        perror("[DEBUG] _getcwd failed");
    }
}

// 画像ベース名
const char* base_image_names[] = {
    "hocho", "kosen", "nagaoka_fireworks", "rice", "ex"
};

// 表示パターン
int frame_durations[] = { 1, 1, 1, 1};  // 各パターンのフレーム数
int num_patterns = 4;

// テクスチャID格納
GLuint normal_texture = 0;
GLuint inv_texture = 0;
GLuint orig_texture = 0;
GLuint texture_sequence[10];  // 最大10パターン

// 画像読み込み
GLuint load_texture(const char* filename, int clip_margin, int clip_enabled) {
    
    printf("[DEBUG] Attempting to load texture: %s\n", filename);

    int width, height, channels;
    unsigned char* data = stbi_load(filename, &width, &height, &channels, 0);

    if (!data) {
        printf("[ERROR] stbi_load failed for: %s\n", filename);
        return 0;
    }

    printf("[DEBUG] stbi_load success: %s (width=%d, height=%d, channels=%d)\n",
           filename, width, height, channels);

    if (clip_enabled && clip_margin > 0) {
        const int upper = 255 - clip_margin;
        const int total = width * height * channels;
        for (int i = 0; i < total; i++) {
            int v = data[i];
            if (v < clip_margin) v = clip_margin;
            else if (v > upper) v = upper;
            data[i] = (unsigned char)v;
        }
    }
    
    GLuint texture_id;
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    
    // テクスチャパラメータ
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    // データ転送
    GLenum format = (channels == 4) ? GL_RGBA : GL_RGB;
    glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);

    
    stbi_image_free(data);
    printf("Loaded: %s (ID: %u)\n", filename, texture_id);
    
    return texture_id;
}

// テクスチャ描画
void draw_texture(GLuint texture_id) {
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

// 高精度タイマー
double get_time_ms() {
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (counter.QuadPart * 1000.0) / frequency.QuadPart;
}

int main(int argc, char** argv) {
    // Windowsタイマー精度を1msに設定
    print_current_directory();
    timeBeginPeriod(1);

    LARGE_INTEGER qpc_frequency;
    QueryPerformanceFrequency(&qpc_frequency);
    const double qpc_to_ms = 1000.0 / (double)qpc_frequency.QuadPart;
    
    // GLFW初期化
    if (!glfwInit()) {
        fprintf(stderr, "GLFW initialization failed\n");
        return -1;
    }

    // ウィンドウ装飾を外し、起動時に最大化
    // glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);

    const char* exe_name = "Image Flicker (C)";
    char title[260];
    if (argc > 0 && argv && argv[0]) {
        const char* p = argv[0];
        const char* s1 = strrchr(p, '/');
        const char* s2 = strrchr(p, '\\');
        const char* base = p;
        if (s1 && s1 > base) base = s1 + 1;
        if (s2 && s2 > base) base = s2 + 1;
        snprintf(title, sizeof(title), "%s", base);
        exe_name = title;
    }

    GLFWwindow* window = glfwCreateWindow(1920, 1080, exe_name, NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    
    glfwMakeContextCurrent(window);
    glfwSwapInterval(INTERVAL);  // 垂直同期

    // 実モニタのリフレッシュレートを表示
    GLFWmonitor* primary = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(primary);
    if (mode) {
        printf("Monitor refresh: %d Hz, interval=%d\n", mode->refreshRate, INTERVAL);
    }
    
    // 画像読み込み
    char filename[256];
    const char* base_name = base_image_names[SELECTED_IMAGE];
    
    snprintf(filename, sizeof(filename), "%s_%d_normal%s.png", 
             base_name,  BRIGHTNESS_DECREASE, STR(COLOR));
    normal_texture = load_texture(filename, CLIP_MARGIN, 0);
    
    snprintf(filename, sizeof(filename), "%s_%d_inv%s.png", 
             base_name, BRIGHTNESS_DECREASE, STR(COLOR));
    inv_texture = load_texture(filename, CLIP_MARGIN, 0);
    
        snprintf(filename, sizeof(filename), "%s.png", base_name);
    orig_texture = load_texture(filename, CLIP_MARGIN, 1);
    
    // テクスチャシーケンス生成（inv → base → normal → base）
    int seq_len = 0;
    texture_sequence[seq_len++] = inv_texture;
            texture_sequence[seq_len++] = orig_texture;
    texture_sequence[seq_len++] = normal_texture;
    texture_sequence[seq_len++] = orig_texture;
    
    printf("Starting render loop...\n");
    
    // メインループ
    int current_index = 0;
    int frame_counter = 0;
    double measurement_start_time = get_time_ms();
    int frames_since_measurement = 0;
    const int TIMING_INTERVAL_FRAMES = 180;
    
    double swap_acc_ms = 0.0;
    double frame_acc_ms = 0.0;
    int swap_samples = 0;
    const int REPORT_INTERVAL = 300;  // ログの頻度を抑えてオーバーヘッド最小化
    LARGE_INTEGER prev_swap = {0};

    while (!glfwWindowShouldClose(window)) {
        draw_texture(texture_sequence[current_index]);
        LARGE_INTEGER pre_swap, post_swap;
        QueryPerformanceCounter(&pre_swap);
        glfwSwapBuffers(window);
        QueryPerformanceCounter(&post_swap);
        glfwPollEvents();
        
        frame_counter++;
        frames_since_measurement++;
        
        if (frame_counter >= frame_durations[current_index]) {
            frame_counter = 0;
            current_index = (current_index + 1) % seq_len;
        }
        
        // 軽量なタイミング計測: swap時間とフレーム間隔の平均を一定間隔でだけ出力
        double swap_ms = (double)(post_swap.QuadPart - pre_swap.QuadPart) * qpc_to_ms;
        swap_acc_ms += swap_ms;
        if (prev_swap.QuadPart != 0) {
            double frame_ms = (double)(post_swap.QuadPart - prev_swap.QuadPart) * qpc_to_ms;
            frame_acc_ms += frame_ms;
        }
        prev_swap = post_swap;

        swap_samples++;
        if (swap_samples >= REPORT_INTERVAL) {
            double avg_swap = swap_acc_ms / (double)swap_samples;
            double avg_frame = frame_acc_ms / (double)(swap_samples - 1 > 0 ? swap_samples - 1 : 1);
            printf("[TIMING] interval=%d avg_swap=%.3fms avg_frame=%.3fms samples=%d\n",
                   INTERVAL, avg_swap, avg_frame, swap_samples);
            swap_acc_ms = 0.0;
            frame_acc_ms = 0.0;
            swap_samples = 0;
        }
    }
    
    // クリーンアップ
    glDeleteTextures(1, &normal_texture);
    glDeleteTextures(1, &inv_texture);
    glDeleteTextures(1, &orig_texture);
    
    glfwDestroyWindow(window);
    glfwTerminate();
    timeEndPeriod(1);
    
    return 0;
}