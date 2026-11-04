"""Image judges: DINOv2 embeddings (QD descriptors) and 'looks like an organism'
(fitness) from a product of experts: CLIP ViT-L/14 and SigLIP 2 (so400m, 384 px)
logits averaged, i.e. their label probabilities multiplied and renormalised. A
form has to convince both at once, which is much harder to game than either:
imitation winners evolved against CLIP alone scored near zero under SigLIP.
The template says "3d render" so that every render, being one, doesn't lose to
the "random 3d shape" decoy on style alone.
"""

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel, AutoProcessor

ORGANISMS = [
    "a brain coral", "a sea anemone", "a lettuce", "a kale leaf", "a nudibranch sea slug",
    "a sea sponge", "a mushroom", "an oyster mushroom", "a lichen", "a cactus",
    "a succulent plant", "a flower", "a rose", "a seashell", "a jellyfish",
    "a fan coral", "a cauliflower", "a walnut", "a human brain", "a pinecone",
    "a seaweed", "a sea urchin", "a morel mushroom", "a cabbage", "a gourd",
    "a staghorn coral", "a lettuce coral", "a chanterelle mushroom", "a bracket fungus",
    "a coral fungus", "a fern",
    # branching plants (tip splitting)
    "a romanesco broccoli", "a broccoli", "a reindeer lichen", "a liverwort", "a fiddlehead fern",
    "a cholla cactus", "a crested coral cactus", "a club moss", "a bonsai tree", "a kelp seaweed",
]
NEGATIVES = [
    "crumpled paper", "a pile of trash", "a smooth blob", "a rock", "a plastic bag",
    "a random 3d shape", "a ball", "a lump of clay", "fabric",
]
TEMPLATE = "a 3d render of {}"
EXPERTS = ("openai/clip-vit-large-patch14", "google/siglip2-so400m-patch14-384")


def _proj(out):
    # transformers>=5 wraps projected features in a ModelOutput's pooler_output.
    return out if torch.is_tensor(out) else out.pooler_output


class Expert:
    """One CLIP-style image/text model."""

    def __init__(self, name, device):
        self.dev = device
        self.proc = AutoProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(device).eval().half()
        # SigLIP was trained on max-length padded text; CLIP's 100 is its own logit scale.
        self.siglip = "siglip" in name
        self.scale = float(self.model.logit_scale.exp()) if self.siglip else 100.0

    @torch.no_grad()
    def text(self, texts):
        pad = dict(padding="max_length", max_length=64) if self.siglip else dict(padding=True)
        tok = self.proc(text=texts, return_tensors="pt", truncation=True, **pad).to(self.dev)
        return torch.nn.functional.normalize(_proj(self.model.get_text_features(**tok)).float(), dim=1)

    @torch.no_grad()
    def image(self, images):
        x = self.proc(images=list(images), return_tensors="pt")["pixel_values"].to(self.dev).half()
        return torch.nn.functional.normalize(_proj(self.model.get_image_features(pixel_values=x)).float(), dim=1)


class Judge:
    def __init__(self, device="cuda", experts=EXPERTS, dino_name="facebook/dinov2-small"):
        self.dev = device
        self.dino_proc = AutoImageProcessor.from_pretrained(dino_name)
        self.dino = AutoModel.from_pretrained(dino_name).to(device).eval().half()
        self.experts = [Expert(n, device) for n in experts]
        self.labels = self.text_embed(ORGANISMS + NEGATIVES)

    def text_embed(self, labels):
        return [e.text([TEMPLATE.format(t) for t in labels]) for e in self.experts]

    def probs(self, images, T):
        """(n_images, n_labels) probabilities from the experts' averaged logits."""
        return torch.stack([e.scale * e.image(images) @ t.T for e, t in zip(self.experts, T)]).mean(0).softmax(1)

    @torch.no_grad()
    def dino_embed(self, images):
        x = self.dino_proc(images=list(images), return_tensors="pt")["pixel_values"].to(self.dev).half()
        h = self.dino(pixel_values=x).last_hidden_state
        # CLS + mean patch token: shape and texture.
        return torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).float().cpu().numpy()

    def score(self, results):
        """Fill descriptors/fitness into evaluate.grow_batch results (alive ones).
        Organism probabilities are means over the three floored `views`, as in imitation."""
        alive = [r for r in results if not r["dead"]]
        if not alive:
            return results
        de = (self.dino_embed([r["iso"] for r in alive]) + self.dino_embed([r["top"] for r in alive])) / 2
        p = self.probs([v for r in alive for v in r["views"]], self.labels)[:, :len(ORGANISMS)]
        p = p.reshape(len(alive), 3, -1).mean(1).cpu().numpy()
        for r, d, pp in zip(alive, de, p):
            k = int(pp.argmax())
            r["dino"] = d
            r["organic"] = float(pp.sum())
            r["label"] = ORGANISMS[k]
            r["label_p"] = float(pp[k])
        return results

    def target_scorer(self, target=None, ref_images=None):
        """Return f(images) -> score in [0,1] for matching a text target or reference photos.

        Text: probability of `target` against the other organisms and the negatives.
        Photos: cosine similarity to the mean reference embedding, mapped to [0,1]
        and averaged over the experts.
        """
        if ref_images is not None:
            refs = [torch.nn.functional.normalize(e.image(ref_images).mean(0), dim=0) for e in self.experts]
            return lambda ims: np.mean([((e.image(ims) @ r).cpu().numpy() - 0.5) * 2
                                        for e, r in zip(self.experts, refs)], 0)
        T = self.text_embed([target] + [o for o in ORGANISMS if o != target] + NEGATIVES)
        return lambda ims: self.probs(ims, T)[:, 0].cpu().numpy()
