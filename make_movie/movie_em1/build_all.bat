@echo off
setlocal enabledelayedexpansion

set SRC=make_movie_2photo.c
set INC=c:\Users\visulab\shu_kondo\Imperceptible-2D-code\vcpkg\installed\x64-mingw-dynamic\include
set LIB=c:\Users\visulab\shu_kondo\Imperceptible-2D-code\vcpkg\installed\x64-mingw-dynamic\lib

rem 出力先フォルダ
if not exist bin mkdir bin

rem base名とインデックスの対応（Cの base_image_names[] と一致させる）
for %%A in (0:hocho 1:kosen 2:nagaoka_fireworks 3:rice 4:ex) do (
  for /f "tokens=1,2 delims=:" %%i in ("%%A") do (
    set IDX=%%i
    set BASENAME=%%j

    for %%X in (1 2 3 4) do (
      for %%C in (R G B I X) do (
        set OUT=bin\!BASENAME!_%%X_%%C.exe

        echo === Building !OUT! (IDX=!IDX!, XX=%%X, COLOR=%%C) ===

        gcc "%SRC%" ^
          -DSELECTED_IMAGE=!IDX! ^
          -DBRIGHTNESS_DECREASE=%%X ^
          -DCOLOR=%%C ^
          -DINTERVAL=1 ^
          -I"%INC%" ^
          -L"%LIB%" ^
          -lglfw3dll -lwinmm -lopengl32 -lgdi32 -luser32 ^
          -o "!OUT!"

        if errorlevel 1 (
          echo [ERROR] Failed: !OUT!
          exit /b 1
        )
      )
    )
  )
)

echo All builds finished. EXEs are in .\bin\
endlocal



