# Breeze Indexing performance

I've timed a few projects to with various embedding providers.
This way you can make a more informed decission on the trade-offs.

## Environment

```shellsession
$ fastfetch
                     ..'          ivan@descartes
                 ,xNMM.           --------------
               .OMMMMo            OS: macOS Sequoia 15.5 arm64
               lMM"               Host: MacBook Pro (16-inch, 2024, Three Thunderbolt 5 ports)
     .;loddo:.  .olloddol;.       Kernel: Darwin 24.5.0
   cKMMMMMMMMMMNWMMMMMMMMMM0:     Uptime: 15 days, 19 hours, 17 mins
 .KMMMMMMMMMMMMMMMMMMMMMMMWd.     Packages: 336 (brew), 50 (brew-cask)
 XMMMMMMMMMMMMMMMMMMMMMMMX.       Shell: zsh 5.9
;MMMMMMMMMMMMMMMMMMMMMMMM:        Display (LG Ultra HD): 5120x2880 @ 60 Hz (as 2560x1440) in 27"
:MMMMMMMMMMMMMMMMMMMMMMMM:        Display (LG HDR 4K): 5120x2880 @ 60 Hz (as 2560x1440) in 27"
.MMMMMMMMMMMMMMMMMMMMMMMMX.       Display (LG Ultra HD): 5120x2880 @ 60 Hz (as 2560x1440) in 27"
 kMMMMMMMMMMMMMMMMMMMMMMMMWd.     DE: Aqua
 'XMMMMMMMMMMMMMMMMMMMMMMMMMMk    WM: Quartz Compositor 278.4.7
  'XMMMMMMMMMMMMMMMMMMMMMMMMK.    WM Theme: Multicolor (Dark)
    kMMMMMMMMMMMMMMMMMMMMMMd      Font: .AppleSystemUIFont [System], Helvetica [User]
     ;KMMMMMMMWXXWMMMMMMMk.       Cursor: Fill - Black, Outline - White (32px)
       "cooc*"    "*coo'"         Terminal: WezTerm 20250320-072107-a8735851
                                  Terminal Font: Monaspace Neon
                                  CPU: Apple M4 Max (16) @ 4.51 GHz
                                  GPU: Apple M4 Max (40) @ 1.58 GHz [Integrated]
                                  Memory: 50.20 GiB / 64.00 GiB (78%)
                                  Swap: 12.74 GiB / 14.00 GiB (91%)
                                  Disk (/): 1.46 TiB / 1.81 TiB (81%) - apfs [Read-only]
                                  Local IP (en0): 192.168.69.250/24
                                  Battery (bq40z651): 80% [AC connected]
                                  Power Adapter: 98W
                                  Locale: en_US.utf-8
```

## Projects

The projects I've used for this experiment:

- [Lance storage engine](https://github.com/lancedb/lance)
- [Lance DB](https://github.com/lancedb/lancedb)
- [Kuzu DB](https://github.com/kuzudb/kuzu)

They represent decently sized projects with a good mix of languages.

### Lance

#### Lance timings

To fully index from scratch:

- **Local (baai/bge-small-en-v1.5)**:
- **Ollama (nomic-embed-text)**: 2m 43s 905ms 639us 625ns
- **Modal (L4 | Qwen3-embed-0.6B)**: 46s 51ms 149us 375ns
- **Voyage**: 46s 73ms 565us 417ns

#### Lance file stats

```shellsession
$ tokei ~/github/lancedb/lance --exclude target --exclude node_modules --exclude .venv

===============================================================================
 Language            Files        Lines         Code     Comments       Blanks
===============================================================================
 Batch                   1           35           26            1            8
 C                       1           87           47           28           12
 CSS                     1            4            4            0            0
 Java                   44         7031         4435         1785          811
 JSON                    3           15           15            0            0
 Makefile                2           75           53            7           15
 Protocol Buffers        8         1804          620          962          222
 Python                142        31448        25314         1400         4734
 ReStructuredText       28         4197         3133            0         1064
 Shell                   3          318          206           79           33
 Plain Text             14           77            0           76            1
 TOML                   21         1650         1312          198          140
 XML                     3          642          585           31           26
 YAML                   10           80           77            2            1
-------------------------------------------------------------------------------
 Jupyter Notebooks       4            0            0            0            0
 |- Markdown             3           85            0           70           15
 |- Python               4          408          339           19           50
 (Total)                            493          339           89           65
-------------------------------------------------------------------------------
 Markdown               28         1206            0          805          401
 |- BASH                 3            7            7            0            0
 |- Java                 1          123          118            2            3
 |- Python               4          115           86            9           20
 |- Rust                 2           30           26            1            3
 |- Shell                5           66           66            0            0
 (Total)                           1547          303          817          427
-------------------------------------------------------------------------------
 Rust                  392       186664       158916         8404        19344
 |- Markdown           299         8659          164         6901         1594
 (Total)                         195323       159080        15305        20938
===============================================================================
 Total                 705       235333       194743        13778        26812
===============================================================================
```

### LanceDB

#### LanceDB timings

To fully index from scratch:

- **Local (baai/bge-small-en-v1.5)**:
- **Ollama (nomic-embed-text)**: 4m 58s 500ms 891us 958ns
- **Modal (L4 | Qwen3-embed-0.6B)**: 2m 15s 698ms 652us 125ns
- **Voyage**: 2m 9s 801ms 186us 542ns

#### LanceDB filestats

```shellsession
$ tokei ~/github/lancedb/lancedb --exclude target --exclude node_modules --exclude .venv

===============================================================================
 Language            Files        Lines         Code     Comments       Blanks
===============================================================================
 CSS                     2          119          102            3           14
 Dockerfile              2           54           27           14           13
 HTML                    2          181          137           31           13
 Java                    2          243          138           77           28
 JavaScript             10          461          296          100           65
 JSON                   30        20851        20850            0            1
 Makefile                1           40           31            0            9
 PowerShell              2           84           58           14           12
 Python                 97        29127        24199         1401         3527
 Shell                   9          396          209          100           87
 SVG                     5           26           26            0            0
 Plain Text              8           37            0           37            0
 TOML                    8          480          434           15           31
 TypeScript             56        17042        11702         3786         1554
 XML                     2          433          415            1           17
 YAML                    3          920          897            3           20
-------------------------------------------------------------------------------
 Jupyter Notebooks      12            0            0            0            0
 |- Markdown            12          684           21          500          163
 |- Python              12         1203          976           48          179
 (Total)                           1887          997          548          342
-------------------------------------------------------------------------------
 Markdown              247        20381            0        12820         7561
 |- BASH                15           80           77            2            1
 |- JavaScript          13          209          181           19            9
 |- JSON                 3           77           77            0            0
 |- Markdown             1          124            0          124            0
 |- Python              71         2683         2150          148          385
 |- Rust                 6           65           63            0            2
 |- Shell               13           77           71            6            0
 |- SQL                  2            6            6            0            0
 |- TOML                 4           17           16            0            1
 |- TypeScript          10          130          122            0            8
 (Total)                          23849         2763        13119         7967
-------------------------------------------------------------------------------
 Rust                   84        25353        21693          632         3028
 |- Markdown            52         2471           94         1924          453
 (Total)                          27824        21787         2556         3481
===============================================================================
 Total                 582       116228        81214        19034        15980
===============================================================================
```

### KuzuDB

#### KuzuDB timings

To fully index from scratch:

- **Local (baai/bge-small-en-v1.5)**:
- **Ollama (nomic-embed-text)**: 14m 30s 299ms 285us 666ns
- **Modal (L4 | Qwen3-embed-0.6B)**: 3m 29s 532ms 523us 875ns
- **Voyage**: 5m 29s 355ms 457us 834ns

#### KuzuDB filestats

```shellsession
$ tokei ~/github/kuzudb/kuzu --exclude target --exclude node_modules --exclude .venv

===============================================================================
 Language            Files        Lines         Code     Comments       Blanks
===============================================================================
 Autoconf               13          562          389           60          113
 BASH                    1            6            6            0            0
 Batch                   1           89           68            0           21
 C                      71        96620        84276         6541         5803
 C Header             1475       299838       225579        42637        31622
 CMake                 242         6581         5374          444          763
 C++                  1145       255272       209912        16062        29298
 C++ Header             22        39621        26875         7757         4989
 Dockerfile              1            5            5            0            0
 Java                   30         4479         3068          819          592
 JavaScript             51         7615         6159          868          588
 JSON                   62        32789        32788            0            1
 Makefile                3          441          334           17           90
 Python                 83        15141        12683          506         1952
 ReStructuredText        1           27           22            0            5
 Shell                  12          383          189          139           55
 SQL                     1          274          135           65           74
 SVG                     1            1            1            0            0
 Plain Text             37        75428            0        75402           26
 TOML                    4          192          165            3           24
 TypeScript              1          393          126          220           47
 YAML                    1           10           10            0            0
-------------------------------------------------------------------------------
 HTML                    2           58           48            0           10
 |- JavaScript           2          229          174           26           29
 (Total)                            287          222           26           39
-------------------------------------------------------------------------------
 Markdown               37         1485            0         1053          432
 |- BASH                 7           23           23            0            0
 |- C                    1           98           65           18           15
 |- C++                  1           33           32            1            0
 |- JavaScript           1            2            2            0            0
 (Total)                           1641          122         1072          447
-------------------------------------------------------------------------------
 Rust                   12         3269         2904          105          260
 |- Markdown             7          309            9          262           38
 (Total)                           3578         2913          367          298
===============================================================================
 Total                3309       840579       611116       152698        76765
===============================================================================
```
