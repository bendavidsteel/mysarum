# Ideas: giving the NCA more control over form

The neural CA (`dgrow/nca.py`) currently controls only the in-plane growth
rate, the anisotropy strength, spontaneous curvature, phototropism and colour.
The growth *direction* comes from two fixed fields (∇ of boundary distance, and
the light). Imitation works for targets that are mounds, plates or ruffles
(brain coral, lettuce coral, sea fan, chanterelle, morel) and fails for
branching ones (staghorn coral, coral fungus) and for forms that need tubes,
stalks or banding (table coral, turkey tail, mushroom coral).

## Tip splitting without a split rule

In corals, plants and lungs, branching is emergent. Tips stick out, so they get
more of a resource (light, nutrient) than the flanks and grow faster; once a
tip is wider than the resource's diffusion length it flattens and splits
(the Mullins–Sekerka instability, as in DLA and Kaandorp's accretive coral
models). Turing spots on a growing domain split the same way. So instead of an
explicit "split here" rule, give the CA two ingredients and let search find the
regime:

1. a growth zone that pushes outward along its own axis, and
2. competition for a resource that makes a broad tip unstable.

## Proposed mechanisms, in order

1. **Polarity from the CA's own gradient.** One channel acts like auxin: the
   anisotropy axis in each face is the intrinsic gradient of that channel,
   blended with the existing ∇d and light axes. A growth zone at a tip then
   elongates along the tip axis. Reuses the per-face gradient code already
   used for ∇d. A second polarity channel gives a second axis, i.e. a full
   CA-defined growth tensor.
2. **Accretion and ambient exposure.** A channel that moves the surface along
   its normal (skeleton deposition, as in coral), and a second exposure cue:
   rays over the whole hemisphere (nutrient access / ambient occlusion), not
   only toward the overhead light. Accretion ∝ exposure is Kaandorp's branching
   rule. Check first with a hand-built preset, no CA, that the physics
   actually branches.
3. **Local material properties.** Per-vertex bending stiffness (stiff ribs in a
   soft membrane: morel ridges, turkey-tail bands, sea-fan veins), and
   spontaneous curvature along vs. across the polarity axis (tubes, funnels,
   scrolls: table-coral stalks, chanterelle trumpets).
4. **CA-controlled smoothing.** The growth field is always blurred
   (`SimConfig.g_smooth`), which caps how sharp a growth zone can be. Let a
   gene or channel set it.
5. **Holes (later).** Delete faces where a channel falls below a threshold:
   lace, sea-fan meshes, and new boundary that can itself grow. Needs topology
   changes in `mesh.py` beyond splits and flips.

## Search side

- Imitation runs were short (10 generations on ~1.2k dimensions); run 30+ for
  targets still improving.
- Use reference photo sets (`imitate.py --refs`) for targets a text prompt
  can't separate from look-alikes (table coral vs. plain plates).
- Add a structural descriptor beside the DINOv2 one (branch count from a
  skeleton, elongation, genus) so QD keeps branching forms before CLIP rewards
  them.

## Plan

1 → 2 → 3 + 4 together; smoke test with a branching preset and random CAs;
then QD and imitation on the coral and fungus targets.
