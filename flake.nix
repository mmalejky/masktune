{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/release-22.05"; # for python 3.9
  };
  outputs = { self, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      fhs = pkgs.buildFHSUserEnv {
        name = "fhs-shell";
        targetPkgs = pkgs: with pkgs; [
          python39Full
          python39Packages.pip
          python39Packages.virtualenv
          python39Packages.tkinter

          git
          gitRepo
          gnupg
          autoconf
          curl
          procps
          gnumake
          util-linux
          m4
          gperf
          unzip
          cudaPackages_11_6.cudatoolkit
          linuxPackages.nvidia_x11
          libGLU libGL
          xorg.libXi xorg.libXmu freeglut
          xorg.libXext xorg.libX11 xorg.libXv xorg.libXrandr zlib 
          ncurses5
          stdenv.cc
          binutils
          glib
        ];
        multiPkgs = pkgs: with pkgs; [ zlib ];
        profile = ''
          export CUDA_PATH=${pkgs.cudatoolkit}
          # export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib
          export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
          export EXTRA_CCFLAGS="-I/usr/include"
          source .venv/bin/activate
        '';
      };
    in { devShells.${system}.default = fhs.env; };
}
