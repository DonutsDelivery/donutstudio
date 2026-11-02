#!/usr/bin/env python3
"""Pins the starter-recipe composite path across the plugin and the helper.

The Gradient and Procedural Shape starters wire their overlay into Blend port 3.
The executor must compile that wiring, validation must agree with the executor,
and both native compositors must draw the gradient. The compile-and-run check
lives in plugin/Tests/VisualRecipeExecutionTests.cpp; this script keeps the
source wiring honest where that test cannot run (the Metal compositor).
"""
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
executor = (root / "src/visual_plan_executor.h").read_text()
snapshot = (root / "src/render_snapshot.h").read_text()
renderer_h = (root / "src/renderer.h").read_text()
renderer = (root / "src/renderer.cpp").read_text()
metal = (root / "src/gpu_backend/backend_sokol.mm").read_text()
recipes = (root.parent / "plugin/Source/VideoVisualRecipe.h").read_text()
recipe_test = (root.parent / "plugin/Tests/VisualRecipeExecutionTests.cpp").read_text()
plugin_cmake = (root.parent / "plugin/CMakeLists.txt").read_text()

# Executor: Draw Shape at either secondary port, and Gradient as an overlay.
assert 'else if (terminalKind == "visual.draw.shape")' in executor
assert 'else if (terminalKind == "visual.gradient")' in executor
assert "execution.drawShapeGradient = true" in executor
assert "drawShapeNodeId = ids[(size_t) terminal]" in executor
assert "layer.drawShapeGradient = execution->drawShapeGradient" in executor
assert 'visual composite secondary branch is not an executable native source' in executor

# Validation names the same reason and admits the same kinds.
assert "isExecutableCompositeSecondaryKind" in snapshot
assert "admitCompositeSecondaryKinds" in snapshot
assert 'kind == "visual.draw.shape" || kind == "visual.gradient"' in snapshot
assert ("visual composite secondary branch is not an executable native source"
        in snapshot)

# The gradient reuses the draw-shape overlay pass in both compositors.
assert "drawShapeGradient" in renderer_h and "drawGradientCos" in renderer_h
assert "uShapeGradient" in renderer and "uShapeColorB" in renderer
assert "mix(uShapeColor, uShapeColorB, t)" in renderer
assert "gradient[4]" in metal and "colorB[4]" in metal
assert "mix(p.color, p.colorB, t)" in metal
size = re.search(r"sizeof \(MetalDrawShapeParams\) == (\d+)", metal)
assert size and int(size.group(1)) == 112, "Metal draw-shape uniform size changed"
# Four float4 rows of the old block plus colorB, gradient and the canvas rect: 7 * 16 bytes.
assert "float4 colorB; float4 gradient; float4 canvasRect;" in metal
# Both backends map RawUV through the canvas: the shape and gradient are authored
# against the canvas, and outside it the export has no pixels at all.
assert "uCanvasRect" in renderer and "uCanvasRect" in renderer_h
assert "vec2 uv = (RawUV - uCanvasRect.xy) / canvasSpan;" in renderer
assert "shapeCanvasRect" in metal

# Recipe wiring: both overlay starters feed Blend port 3 and carry visible values.
assert "{ gradient, 0 }, { blend, 3 }" in recipes
assert "{ draw, 1 }, { blend, 3 }" in recipes
assert "setStarterColour(graph, gradient" in recipes
assert "setStarterParameters(graph, shape" in recipes

# The execution test exists, covers every built-in recipe, and is registered.
for factory in ("makeEditableLayerRecipe", "makeStarterGradientRecipe",
                "makeStarterShapeRecipe", "makeStarterFeedbackRecipe",
                "makeStarterImportedSceneRecipe"):
    assert factory in recipe_test, factory
assert "compileVisualLayerExecution" in recipe_test
assert "Tests/VisualRecipeExecutionTests.cpp" in plugin_cmake
# A new built-in recipe factory must be added to the execution test.
factories = set(re.findall(r"inline Recipe (makeStarter\w+Recipe|makeEditableLayerRecipe)\s*\(", recipes))
for factory in factories:
    assert factory in recipe_test, f"{factory} is not covered by VisualRecipeExecutionTests.cpp"

print("visual recipe composite source contract: PASS")
