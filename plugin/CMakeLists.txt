# Renderer target registration consumed by the helper geometry check.
        add_executable(CustomShaderAuthorityPixelAcceptance
            Tests/CustomShaderAuthorityPixelAcceptance.cpp
            ../video-helper/src/renderer.cpp
            ../video-helper/src/geometry_core_backend.cpp
            ../video-helper/src/fixture_scene_renderer.cpp
            ../video-helper/src/color_transform_gl.cpp
            ../video-helper/src/gpu_backend/backend_opengl.cpp
            ../video-helper/src/sdf_native_renderer.cpp
            ../video-helper/src/deformation_contract.cpp
            ../video-helper/src/native_animation_deformation.cpp
            ../video-helper/src/shader_generator.cpp
            ../video-helper/src/particle_engine.cpp
            ../video-helper/src/gl_loader.cpp
            ../video-helper/src/lut_loader.cpp)
