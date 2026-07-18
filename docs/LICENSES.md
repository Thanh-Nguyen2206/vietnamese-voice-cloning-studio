# License review

Kiểm tra ngày 2026-07-18; đây không phải tư vấn pháp lý. Luôn đọc lại model card/license trước khi phát hành.

| Thành phần | License ghi nhận | Hệ quả chính |
|---|---|---|
| F5-TTS source code | MIT | Code permissive; không thay thế license checkpoint/data. |
| `hynt/F5-TTS-Vietnamese-ViVoice` | CC-BY-NC-SA-4.0 | Phi thương mại, attribution/share-alike theo điều khoản model. |
| F5 pretrained upstream | CC-BY-NC | Phi thương mại do dữ liệu huấn luyện upstream. |
| `capleaf/viXTTS` | Coqui Public Model License | Model và output chỉ dùng phi thương mại theo model card. |
| `facebook/mms-tts-vie` | CC-BY-NC-4.0 | Phi thương mại. |
| Bark source/checkpoint | MIT | Project nói checkpoint sẵn sàng cho commercial use; vẫn review prompt/voice rights. |
| Piper legacy source / `piper-tts` hiện dùng | legacy MIT / package GPL-3.0-or-later | Docker phân phối package GPL; cần tuân thủ notice/source obligations. |
| Piper `vais1000` voice | dataset CC-BY-4.0; fine-tuned từ lessac | Model card yêu cầu attribution; nguồn lessac làm license lineage cần legal review trước commercial use. |
| `edge-tts` client | LGPLv3 (trừ helper MIT) | Client license khác điều khoản dịch vụ Microsoft; cần review service terms riêng. |

Nguồn chính: [ViVoice model card](https://huggingface.co/hynt/F5-TTS-Vietnamese-ViVoice),
[F5-TTS repository/license](https://github.com/SWivid/F5-TTS),
[viXTTS license](https://huggingface.co/capleaf/viXTTS/blob/main/LICENSE.txt),
[MMS Vietnamese model card](https://huggingface.co/facebook/mms-tts-vie),
[Bark repository](https://github.com/suno-ai/bark),
[Piper voices guidance](https://github.com/rhasspy/piper),
[vais1000 model card](https://huggingface.co/rhasspy/piper-voices/blob/main/vi/vi_VN/vais1000/medium/MODEL_CARD),
[edge-tts license](https://github.com/rany2/edge-tts/blob/master/LICENSE).

Do ViVoice, viXTTS và MMS đều có hạn chế phi thương mại, toàn bộ demo mặc định được mô tả là nghiên cứu/
giáo dục. Một hướng thương mại phải thay toàn bộ checkpoint bị hạn chế, xác minh dataset/voice lineage,
đánh giá license dependency khi phân phối và xin tư vấn pháp lý phù hợp.
