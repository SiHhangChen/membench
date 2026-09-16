# WR-07

WR-07 is a single-item color-memory transport task. Three sugar plates are
placed on the coffee-machine side and three matching-color sugar plates are
placed on the opposite counter. The three colors are sampled without
replacement from the 20-color pool and both side assignments are deterministic
for the episode seed. The target assignment is a non-identity permutation, so
the matching plate is not inferred from its left/center/right position.

The expert observes the source plates, picks the milk carton from one source
plate, returns the arm to its regular holding pose, latches the source color,
places the carton on the matching target plate, verifies the placement, and
returns the arm to its regular empty pose. Source and target in-plate XY
offsets and yaw are seed-owned and bounded by the configured 4.5 cm radius.

The task uses the fixed, stable `milk_1` carton geometry. Its appearance is
sampled from four texture-and-finish profiles: kraft-paper matte, light-green
satin, cool-blue glossy, and silver-blue foil. The green packaging region of
the original texture is recolored while its white artwork and text remain
intact. Thus each seed selects one of four clearly visible appearance variants
without changing the carton collision geometry.

The source pickup uses one shared calibrated ``counter_pick_station`` sampled
uniformly from a 10 cm-diameter disk, with a small calibrated yaw variation;
the sampled pose is reused by the observe and pick stages. The target side uses
one calibrated 10 cm-diameter disk sampled at navigation time as well.

The initial base XY is sampled uniformly in area from a 5 cm-radius disk
centered on the environment reset pose. Initial yaw is preserved. This draw
uses an independent seed salt and does not alter source/target station draws.
Initialization metadata records the center, sampled XY, offset and radius.

During held transport the main in-place turn is sampled around the calibrated
signed turn of -178.84 degrees with an independent uniform +/-5 degree offset.
The primitive keeps the clockwise branch, including draws slightly beyond
-180 degrees. The target translation then ends with a small yaw correction
when required.
