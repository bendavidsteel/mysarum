"""Image judges: DINOv2 embeddings (descriptors) and CLIP 'looks like an organism' (fitness)."""

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

ORGANISMS = [
    "a brain coral", "a sea anemone", "a lettuce", "a kale leaf", "a nudibranch sea slug",
    "a sea sponge", "a mushroom", "an oyster mushroom", "a lichen", "a cactus",
    "a succulent plant", "a flower", "a rose", "a seashell", "a jellyfish",
    "a fan coral", "a cauliflower", "a walnut", "a human brain", "a pinecone",
    "a seaweed", "a sea urchin", "a morel mushroom", "a cabbage", "a gourd",
]
NEGATIVES = [
    "crumpled paper", "a pile of trash", "a smooth blob", "a rock", "a plastic bag",
    "a random 3d shape", "a ball", "a lump of clay", "fabric",
]
TEMPLATE = "a photo of {}"


def _proj(out):
    # transformers>=5 wraps projected features in a ModelOutput's pooler_output.
    return out if torch.is_tensor(out) else out.pooler_output


class Judge:
    def __init__(self, device="cuda", clip_name="openai/clip-vit-large-patch14",
                 dino_name="facebook/dinov2-small"):
        self.dev = device
        self.dino_proc = AutoImageProcessor.from_pretrained(dino_name)
        self.dino = AutoModel.from_pretrained(dino_name).to(device).eval().half()
        self.clip_proc = CLIPProcessor.from_pretrained(clip_name)
        self.clip = CLIPModel.from_pretrained(clip_name).to(device).eval().half()
        texts = [TEMPLATE.format(t) for t in ORGANISMS + NEGATIVES]
        with torch.no_grad():
            tok = self.clip_proc(text=texts, return_tensors="pt", padding=True).to(device)
            t = _proj(self.clip.get_text_features(**tok))
        self.text = torch.nn.functional.normalize(t.float(), dim=1)

    @torch.no_grad()
    def dino_embed(self, images):
        x = self.dino_proc(images=list(images), return_tensors="pt")["pixel_values"].to(self.dev).half()
        h = self.dino(pixel_values=x).last_hidden_state
        # CLS + mean patch token: shape and texture.
        return torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).float().cpu().numpy()

    @torch.no_grad()
    def clip_scores(self, images):
        """(organism probability, best organism label, image embedding) per image."""
        x = self.clip_proc(images=list(images), return_tensors="pt")["pixel_values"].to(self.dev).half()
        e = torch.nn.functional.normalize(_proj(self.clip.get_image_features(pixel_values=x)).float(), dim=1)
        logits = 100.0 * e @ self.text.T
        p = logits.softmax(1)
        k = len(ORGANISMS)
        org = p[:, :k]
        return org.sum(1).cpu().numpy(), [ORGANISMS[i] for i in org.argmax(1).cpu().numpy()], \
            org.max(1).values.cpu().numpy(), e.cpu().numpy()

    def score(self, results):
        """Fill descriptors/fitness into evaluate.grow_batch results (alive ones)."""
        alive = [r for r in results if not r["dead"]]
        if not alive:
            return results
        iso = [r["iso"] for r in alive]
        top = [r["top"] for r in alive]
        de = (self.dino_embed(iso) + self.dino_embed(top)) / 2
        p_iso, lab, pmax, ce = self.clip_scores(iso)
        p_top, _, _, _ = self.clip_scores(top)
        for r, d, a, b, l, m, c in zip(alive, de, p_iso, p_top, lab, pmax, ce):
            r["dino"] = d
            r["clip_emb"] = c
            r["organic"] = float(max(a, b))
            r["label"] = l
            r["label_p"] = float(m)
        return results

    @torch.no_grad()
    def clip_image_embed(self, images):
        x = self.clip_proc(images=list(images), return_tensors="pt")["pixel_values"].to(self.dev).half()
        return torch.nn.functional.normalize(_proj(self.clip.get_image_features(pixel_values=x)).float(), dim=1)

    @torch.no_grad()
    def target_scorer(self, target=None, ref_images=None):
        """Return f(images) -> score in [0,1] for matching a text target or reference photos.

        Text: probability of `target` against the other organisms and the negatives.
        Photos: cosine similarity to the mean reference embedding, mapped to [0,1].
        """
        if ref_images is not None:
            ref = torch.nn.functional.normalize(self.clip_image_embed(ref_images).mean(0), dim=0)
            return lambda ims: ((self.clip_image_embed(ims) @ ref).cpu().numpy() - 0.5) * 2
        others = [o for o in ORGANISMS if o != target] + NEGATIVES
        texts = [TEMPLATE.format(t) for t in [target] + others]
        tok = self.clip_proc(text=texts, return_tensors="pt", padding=True).to(self.dev)
        T = torch.nn.functional.normalize(_proj(self.clip.get_text_features(**tok)).float(), dim=1)
        return lambda ims: (100.0 * self.clip_image_embed(ims) @ T.T).softmax(1)[:, 0].cpu().numpy()
