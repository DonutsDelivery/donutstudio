class Glfw < Formula
  desc "Multi-platform library for OpenGL applications"
  homepage "https://www.glfw.org/"
  url "https://github.com/glfw/glfw/archive/refs/tags/3.5.1.tar.gz"
  sha256 "5234f4f29473e9a06bc7847d8371858dd135d38466eeeaa652fdc9f8f9ff0c20"
  license "Zlib"
  compatibility_version 1
  head "https://github.com/glfw/glfw.git", branch: "master"

  depends_on "cmake" => :build
  depends_on "pkgconf" => :build

  on_linux do
    depends_on "xorg-server" => :test
    depends_on "freeglut"
    depends_on "libxcursor"
    depends_on "libxext"
    depends_on "libxi"
    depends_on "libxinerama"
    depends_on "libxkbcommon"
    depends_on "mesa"
  end

  def install
    system "cmake", "-S", ".", "-B", "build", *std_cmake_args
    system "cmake", "--build", "build"
    lib.install "build/src/libglfw3.a"

    system "cmake", "-S", ".", "-B", "build", "-DBUILD_SHARED_LIBS=TRUE", *std_cmake_args
    system "cmake", "--build", "build"
    system "cmake", "--install", "build"
  end

  test do
    (testpath/"test.c").write <<~C
      #define GLFW_INCLUDE_GLU
      #include <GLFW/glfw3.h>
      #include <stdlib.h>
      int main()
      {
        if (!glfwInit())
          exit(EXIT_FAILURE);
        glfwTerminate();
        return 0;
      }
    C

    system ENV.cc, "test.c", "-o", "test", "-I#{include}", "-L#{lib}", "-lglfw"
    if OS.linux? && ENV.exclude?("DISPLAY")
      system Formula["xorg-server"].bin/"xvfb-run", "./test"
    else
      system "./test"
    end
  end
end
