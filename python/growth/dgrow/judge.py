"""Image judge: SigLIP 2 (so400m, 384 px) gives both the QD descriptors (image
embeddings) and 'looks like an organism' (fitness, imitation targets) as label
probabilities against organisms and decoys. The template says "3d render" so
that every render, being one, doesn't lose to the "random 3d shape" decoy on
style alone.
"""

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

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
MODEL = "google/siglip2-so400m-patch14-384"


def _proj(out):
    # transformers>=5 wraps projected features in a ModelOutput's pooler_output.
    return out if torch.is_tensor(out) else out.pooler_output


class Judge:
    def __init__(self, device="cuda", name=MODEL):
        self.dev = device
        self.proc = AutoProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(device).eval().half()
        self.scale = float(self.model.logit_scale.exp())
        self.labels = self.text_embed(ORGANISMS + NEGATIVES)

    @torch.no_grad()
    def text_embed(self, labels):
        # SigLIP was trained on max-length padded text.
        tok = self.proc(text=[TEMPLATE.format(t) for t in labels], return_tensors="pt", truncation=True,
                        padding="max_length", max_length=64).to(self.dev)
        return torch.nn.functional.normalize(_proj(self.model.get_text_features(**tok)).float(), dim=1)

    @torch.no_grad()
    def image(self, images):
        x = self.proc(images=list(images), return_tensors="pt")["pixel_values"].to(self.dev).half()
        return torch.nn.functional.normalize(_proj(self.model.get_image_features(pixel_values=x)).float(), dim=1)

    def probs(self, images, T):
        """(n_images, n_labels) label probabilities."""
        return (self.scale * self.image(images) @ T.T).softmax(1)

    def score(self, results):
        """Fill descriptors/fitness into evaluate.grow_batch results (alive ones).
        Embedding = mean of the angled and top views; organism probabilities are
        means over the three floored `views`, as in imitation."""
        alive = [r for r in results if not r["dead"]]
        if not alive:
            return results
        de = ((self.image([r["iso"] for r in alive]) + self.image([r["top"] for r in alive])) / 2).cpu().numpy()
        p = self.probs([v for r in alive for v in r["views"]], self.labels)[:, :len(ORGANISMS)]
        p = p.reshape(len(alive), 3, -1).mean(1).cpu().numpy()
        for r, d, pp in zip(alive, de, p):
            k = int(pp.argmax())
            r["emb"] = d
            r["organic"] = float(pp.sum())
            r["label"] = ORGANISMS[k]
            r["label_p"] = float(pp[k])
        return results

    def target_scorer(self, target=None, ref_images=None):
        """Return f(images) -> score in [0,1] for matching a text target or reference photos.

        Text: probability of `target` against the other organisms and the negatives.
        Photos: cosine similarity to the mean reference embedding, mapped to [0,1].
        """
        if ref_images is not None:
            ref = torch.nn.functional.normalize(self.image(ref_images).mean(0), dim=0)
            return lambda ims: ((self.image(ims) @ ref).cpu().numpy() - 0.5) * 2
        T = self.text_embed([target] + [o for o in ORGANISMS if o != target] + NEGATIVES)
        return lambda ims: self.probs(ims, T)[:, 0].cpu().numpy()
