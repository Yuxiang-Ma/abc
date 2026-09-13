# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import sentencepiece


def _assert_file_exists(model_path: str):
    assert os.path.isfile(model_path), model_path

_BEGIN_IMAGE_TOKEN = 255999
_END_IMAGE_TOKEN = 256000
_STATE_TOKEN_TAG = "<state>"

class Tokenizer:

    def __init__(self, model_path: str | None):
        _assert_file_exists(model_path)
        self.sp_model = sentencepiece.SentencePieceProcessor()
        self.sp_model.Load(model_path)

        # BOS / EOS token IDs.
        self.n_words: int = self.sp_model.GetPieceSize()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        self.boi_id: int = _BEGIN_IMAGE_TOKEN
        self.eoi_id: int = _END_IMAGE_TOKEN
        # IMPORTANT: image placeholder token must NOT share ID with pad_id, otherwise
        # padding gets mistaken for image placeholders during populate_image_embeddings().
        # We reserve one "<unused...>" token for this purpose.
        self.image_token_placeholder_id: int = -1
        # Back-compat: state token remains a first-class concept.
        self.state_token_tag: str = _STATE_TOKEN_TAG
        self.state_proj_id: int | None = None

        # Generic special token tags, e.g. "<value>" markers.
        self._special_tag_to_id: dict[str, int] = {}
        self.unused_ids = [
            i for i in range(self.sp_model.GetPieceSize())
            if self.sp_model.IdToPiece(i).startswith("<unused")
        ]
        assert len(self.unused_ids) > 0, "No unused token IDs remaining in tokenizer vocab."
        # Reserve a dedicated placeholder ID for image patch positions.
        self.image_token_placeholder_id = int(self.unused_ids.pop(0))

    def register_special_token(self, token_id: int, tag: str):
        """Register a placeholder tag that should encode to a fixed token id."""
        assert isinstance(tag, str) and len(tag) > 0
        self._special_tag_to_id[tag] = int(token_id)

    def register_state_token(self, token_id: int, tag: str | None = None):
        """Configure a dedicated token id and tag used for state placeholders."""
        self.state_proj_id = token_id
        if tag is not None:
            self.state_token_tag = tag
        if self.state_proj_id is not None:
            self.register_special_token(self.state_proj_id, self.state_token_tag)

    def encode(self, s: str, bos: bool = True, eos: bool = False) -> list[int]:
        """Converts a string into a list of tokens."""
        assert isinstance(s, str)
        # Encode while preserving any registered special tags by inserting their token IDs.
        if self._special_tag_to_id:
            t: list[int] = []
            pos = 0
            tags: list[tuple[str, int]] = list(self._special_tag_to_id.items())
            while pos < len(s):
                next_idx: int | None = None
                next_tag: str | None = None
                next_id: int | None = None
                for tag, tok_id in tags:
                    i = s.find(tag, pos)
                    if i == -1:
                        continue
                    if next_idx is None or i < next_idx:
                        next_idx = i
                        next_tag = tag
                        next_id = tok_id

                if next_idx is None:
                    if pos < len(s):
                        t.extend(self.sp_model.EncodeAsIds(s[pos:]))
                    break

                if next_idx > pos:
                    t.extend(self.sp_model.EncodeAsIds(s[pos:next_idx]))
                assert next_tag is not None and next_id is not None
                t.append(int(next_id))
                pos = next_idx + len(next_tag)
        else:
            t = self.sp_model.EncodeAsIds(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def get_unused_id(self) -> int:
        assert len(self.unused_ids) > 0, "No unused token IDs remaining in tokenizer vocab."
        return self.unused_ids.pop(0)