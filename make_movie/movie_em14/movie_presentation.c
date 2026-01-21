// 実行方法
// gcc make_movie_2photo.c   -I"c:\Users\visulab\shu_kondo\Imperceptible-2D-code\vcpkg\installed\x64-mingw-dynamic\include"  -L"c:\Users\visulab\shu_kondo\Imperceptible-2D-code\vcpkg\installed\x64-mingw-dynamic\lib"   -lglfw3dll -lwinmm -lopengl32 -lgdi32 -luser32 -o make_movie_2photo.exe

#include <GLFW/glfw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <time.h>
#include <direct.h> 

#ifndef SELECTED_IMAGE
#define SELECTED_IMAGE 4
#endif

#ifndef BRIGHTNESS_DECREASE
#define BRIGHTNESS_DECREASE 20
#endif

#ifndef INTERVAL
#define INTERVAL 1
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
int frame_durations[] = { 1, 1};  // 各パターンのフレーム数
int num_patterns = 2;

// テクスチャID格納
GLuint normal_texture = 0;
GLuint inv_texture = 0;
GLuint orig_texture = 0;
GLuint texture_sequence[10];  // 最大10パターン

// 画像読み込み
GLuint load_texture(const char* filename) {
    
    printf("[DEBUG] Attempting to load texture: %s\n", filename);

    int width, height, channels;
    unsigned char* data = stbi_load(filename, &width, &height, &channels, 0);

    if (!data) {
        printf("[ERROR] stbi_load failed for: %s\n", filename);
        return 0;
    }

    printf("[DEBUG] stbi_load success: %s (width=%d, height=%d, channels=%d)\n",
           filename, width, height, channels);
    
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

int main() {
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

    // ========== 変更点: ここからフルスクリーン設定 ==========
    
    // プライマリモニタを取得
    GLFWmonitor* primary = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(primary);

    // ヒント設定：モニタのリフレッシュレートと色深度に合わせる
    glfwWindowHint(GLFW_RED_BITS, mode->redBits);
    glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
    glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
    glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
    
    // フルスクリーンで作成 (第4引数に monitor を指定)
    // 解像度もモニタに合わせることで、ドットバイドット表示になり補間を防ぐ
    GLFWwindow* window = glfwCreateWindow(mode->width, mode->height, "Image Flicker (C)", primary, NULL);
    
    // ========== 変更点: ここまで ==========

    if (!window) {
        printf("[ERROR] Failed to create fullscreen window.\n");
        glfwTerminate();
        return -1;
    }
    
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);  // 手動でフレーム同期するため vsync 無効

    // 実モニタのリフレッシュレートを表示
    int refresh_hz = 60;
    if (mode) {
        refresh_hz = mode->refreshRate;
        printf("Monitor refresh: %d Hz, interval=%d\n", mode->refreshRate, INTERVAL);
        printf("Resolution: %d x %d\n", mode->width, mode->height);
    }

    // INTERVAL 倍のフレーム期間をミリ秒に換算（vsync に依存しない）
    const double target_frame_ms = (1000.0 / (double)refresh_hz) * (double)INTERVAL;
    printf("Target frame: %.3f ms (manual pacing)\n", target_frame_ms);
    
    // 画像読み込み
    char filename[256];
    const char* base_name = base_image_names[SELECTED_IMAGE];
    
    snprintf(filename, sizeof(filename), "%s_%d_normal%s.png", 
             base_name,  BRIGHTNESS_DECREASE, STR(COLOR));
    normal_texture = load_texture(filename);
    
    snprintf(filename, sizeof(filename), "%s_%d_inv%s.png", 
             base_name, BRIGHTNESS_DECREASE, STR(COLOR));
    inv_texture = load_texture(filename);
    
    // テクスチャシーケンス生成
    int is_normal = 1;
    int seq_len = 0;
    for (int i = 0; i < num_patterns; i++) {
        if (frame_durations[i] == 1) {
            texture_sequence[seq_len++] = is_normal ? normal_texture : inv_texture;
            is_normal = !is_normal;
        } else {
            texture_sequence[seq_len++] = orig_texture;
        }
    }
    
    printf("Starting render loop... Press ESC to exit.\n");
    
    // メインループ
    int current_index = 0;
    int frame_counter = 0;
    double swap_acc_ms = 0.0;
    double frame_acc_ms = 0.0;
    int swap_samples = 0;
    const int REPORT_INTERVAL = 300;
    LARGE_INTEGER prev_swap = {0};

    // マウスカーソルを隠す（お好みで）
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);

    while (!glfwWindowShouldClose(window)) {
        double frame_start_ms = get_time_ms();
        // エスケープキーで終了
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, 1);
        }

        draw_texture(texture_sequence[current_index]);
        
        LARGE_INTEGER pre_swap, post_swap;
        QueryPerformanceCounter(&pre_swap);
        glfwSwapBuffers(window);
        QueryPerformanceCounter(&post_swap);
        glfwPollEvents();
        
        frame_counter++;
        
        if (frame_counter >= frame_durations[current_index]) {
            frame_counter = 0;
            current_index = (current_index + 1) % seq_len;
        }
        
        // タイミング計測ログ
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

        // vsync なしでも周期を保つ手動スリープ
        double elapsed_ms = get_time_ms() - frame_start_ms;
        double remain_ms = target_frame_ms - elapsed_ms;
        if (remain_ms > 0) {
            Sleep((DWORD)remain_ms);
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