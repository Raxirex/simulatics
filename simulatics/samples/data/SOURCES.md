# Reference clips: sources and terms

The `.npz` files in this folder are **not** covered by the project's MIT licence. They are derived from the
CMU Graphics Lab Motion Capture Database and stay under its terms.

> The data used in this project was obtained from mocap.cs.cmu.edu.
> The database was created with funding from NSF EIA-0196217.

**Terms** (http://mocap.cs.cmu.edu/, home page and FAQ): the data is free for all uses, may be copied,
modified and redistributed without permission, and may be included in commercially sold products, but the
motion data itself may not be resold directly, even in converted form.

**What was done to it.** The BVH conversion is by Bruce Hahne (cgspeed.com, 2010 release), fetched from
github.com/una-dinosauria/cmu-mocap; he adds no restrictions beyond CMU's. Each take was trimmed, its frame
rate corrected where the file header is wrong (subjects 88 and 89 were captured at 60 fps), retargeted to the
Simulatics humanoid (different proportions and joint limits), cleaned so nothing passes through the floor,
and stored as joint trajectories at 50 fps. `front_flip.npz` is the backflip take reversed in time. The
original files are not stored here; `python -m simulatics.samples.build` downloads them and rebuilds
everything.

| File | Sample | CMU take | Used |
|---|---|---|---|
| `backflip.npz` | Backflip | subject 88 trial 1 | whole take |
| `front_flip.npz` | Front flip (reversed backflip) | subject 88 trial 1 | whole take |
| `front_handspring.npz` | Front handspring | subject 90 trial 11 | whole take |
| `one_hand_stand.npz` | Stand on one hand | subject 89 trial 3 | whole take |
| `handstand.npz` | Handstand | subject 85 trial 5 | frames 400-900 |
| `cartwheel.npz` | Cartwheel | subject 88 trial 7 | whole take |
| `box_jump_up.npz` | Jump onto a box | subject 88 trial 2 | frames 96-200 |
| `box_jump_down.npz` | Jump down from a box | subject 82 trial 4 | frames 150-420 |
| `hurdle.npz` | Jump a hurdle | subject 127 trial 25 | whole take |
| `vertical_jump.npz` | Vertical jump | subject 16 trial 3 | whole take |
| `broad_jump.npz` | Broad jump | subject 13 trial 11 | whole take |

## Recorded motor commands (`*.tape.npz`)

The `.tape.npz` files are motor commands found by tracking the clips above in the physics simulation
(`python -m simulatics.samples.build_physical`). They are derived from the same CMU takes and stay under the
same terms. `physical.json` lists which clip each one follows and what was measured when it was played.
