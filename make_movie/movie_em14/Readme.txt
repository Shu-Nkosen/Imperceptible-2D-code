exe大量作成コマンド

2,4,6,8

& {
  $src = "C:/Users/visulab/shu_kondo/Imperceptible-2D-code/make_movie/movie_em14/make_movie_2photo.c"
  $inc = "C:/Users/visulab/shu_kondo/Imperceptible-2D-code/vcpkg/installed/x64-mingw-dynamic/include"
  $lib = "C:/Users/visulab/shu_kondo/Imperceptible-2D-code/vcpkg/installed/x64-mingw-dynamic/lib"

  $idxMap = @{ ex=4; nagaoka=2; hocho=0; rice=3 }
  $bases  = @("ex","nagaoka","hocho","rice")
  $colors = @("R","G","B","I","X")
  $decs   = @(2, 4, 6 ,8)

  foreach ($b in $bases) {
    $idx = $idxMap[$b]
    foreach ($c in $colors) {
      foreach ($d in $decs) {
        $out = "${b}_${c}${d}.exe"
        $args = @(
          $src,
          "-DSELECTED_IMAGE=$($idx)",
          "-DBRIGHTNESS_DECREASE=$($d)",
          "-DCOLOR=$($c)",
          "-I$inc", "-L$lib",
          "-lglfw3dll", "-lwinmm", "-lopengl32", "-lgdi32", "-luser32",
          "-o", $out
        )
        gcc @args
      }
    }
  }
}


1-4

& {
  $src = "C:/Users/visulab/shu_kondo/Imperceptible-2D-code/make_movie/movie_em14/make_movie_2photo.c"
  $inc = "C:/Users/visulab/shu_kondo/Imperceptible-2D-code/vcpkg/installed/x64-mingw-dynamic/include"
  $lib = "C:/Users/visulab/shu_kondo/Imperceptible-2D-code/vcpkg/installed/x64-mingw-dynamic/lib"

  $idxMap = @{ ex=4; nagaoka=2; hocho=0; rice=3 }
  $bases  = @("ex","nagaoka","hocho","rice")
  $colors = @("R","G","B","I","X")
  $decs   = 1..4

  foreach ($b in $bases) {
    $idx = $idxMap[$b]
    foreach ($c in $colors) {
      foreach ($d in $decs) {
        $out = "${b}_${c}${d}.exe"
        $args = @(
          $src,
          "-DSELECTED_IMAGE=$($idx)",
          "-DBRIGHTNESS_DECREASE=$($d)",
          "-DCOLOR=$($c)",
          "-I$inc", "-L$lib",
          "-lglfw3dll", "-lwinmm", "-lopengl32", "-lgdi32", "-luser32",
          "-o", $out
        )
        gcc @args
      }
    }
  }
}

