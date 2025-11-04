import torch
import torch.nn.functional as F

from torch import nn
from torchvision import transforms

images_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
criterion = torch.nn.KLDivLoss(reduction='batchmean')
class RPA_uap:
    def __init__(self, model, ref_model, step, device, args):
        self.model = model
        self.ref_model = ref_model
        self.step = step
        self.device = device
        self.args = args
        self.prompt = 'a picture of '
        self.prompt_length = len(self.model.tokenizer(self.prompt).input_ids) - 1

    def generate_uap(self, image_batch, images, uap, texts):
        uap = uap.clone().detach().requires_grad_(True)
        uap = uap.to(device=self.device)
        adv_images = torch.clamp(images + uap, 0, 1)
        adv_images = images_normalize(adv_images)

        loss = self._compute_recursive_caption_loss(adv_images, images, texts)

        # Calculate gradients using backward()
        loss.backward()
        grad = uap.grad
        with torch.no_grad():
            uap = self.step.step(uap, grad)
            uap = self.step.project(uap)

        adv_images = image_batch + uap
        adv_images = torch.clamp(adv_images, 0, 1)

        return adv_images, uap

    def _compute_recursive_caption_loss(self, adv_images, images, texts):
        adv_embeds = self.model.visual_encoder(adv_images)
        ori_embeds = self.model.visual_encoder(images)

        adv_att = torch.ones(adv_embeds.size()[:-1], dtype=torch.long).to(self.device)

        output_captions = self._perturb_embedding(adv_embeds)

        if len(adv_embeds) != self.text_embeds_len:
            loss_kl = self._compute_language_modeling_loss(output_captions, adv_embeds, adv_att, images, texts)
        else:
            loss_kl = self._compute_kl_loss(texts, adv_embeds, self.ori_embed)

        text_embeds = self.ref_model.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        text_output = self.ref_model.text_encoder(text_embeds.input_ids, attention_mask=text_embeds.attention_mask,
                                                return_dict=True, mode='text').last_hidden_state
        text_output = text_output[:, 0, :].detach()

        loss_kl_txt = criterion(adv_embeds[:, 0, :].log_softmax(dim=-1), text_output.softmax(dim=-1))

        ori_embeds = ori_embeds.flatten(1).detach()
        loss_kl_ori = criterion(adv_embeds.flatten(1).log_softmax(dim=-1), ori_embeds.softmax(dim=-1))

        total_loss = - self.args.alpha1 * loss_kl - self.args.alpha2 * loss_kl_ori - self.args.alpha3 * loss_kl_txt

        return total_loss

    def _perturb_embedding(self, image_embeds, max_length=30):
        input_ids = torch.tensor([self.model.tokenizer.bos_token_id]).unsqueeze(0).to(self.device)

        adv_caption = []
        bs = image_embeds.size(0)
        adv_caption = [[] for _ in range(bs)]

        for step in range(max_length):
            adv_embeds = image_embeds + self.args.beta * torch.randn_like(image_embeds).to(self.device)

            image_atts = torch.ones(adv_embeds.size()[:-1], dtype=torch.long).to(self.device)

            logits = self.model.text_decoder(input_ids=input_ids,
                                             encoder_hidden_states=adv_embeds,
                                             encoder_attention_mask=image_atts,
                                             return_dict=True,
                                             ).logits

            eos_token_id = self.model.tokenizer.eos_token_id
            filtered_logits = self.top_k_top_p_filtering(logits[:, -1, :])

            vocab_size = len(self.model.tokenizer)
            filtered_logits[:, vocab_size:] = -float('inf')

            next_token_id = self.multinomial(F.softmax(filtered_logits, dim=-1), 1)

            for i in range(bs):
                adv_caption[i].append(next_token_id[i].item())
                input_ids = torch.cat([input_ids, next_token_id[i].unsqueeze(0)], dim=1)

                if next_token_id[i].item() == self.model.tokenizer.eos_token_id:
                    break

        output_text = [self.model.tokenizer.decode(caption, skip_special_tokens=True) for caption in adv_caption]
        output_texts = [text.strip() for text in output_text]

        return output_texts

    def top_k_top_p_filtering(self, logits, filter_value=-float('Inf')):
        """ Filter a distribution of logits using nucleus (top-p) sampling """
        sorted_logits, sorted_indices = torch.sort(logits, descending=False)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > self.args.top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

        logits[indices_to_remove] = filter_value
        return logits

    def _compute_language_modeling_loss(self, adv_captions, adv_embeds, adv_atts, image, texts):
        prompt_length = len(self.model.tokenizer(self.prompt).input_ids) - 1

        image_embed = self.model.visual_encoder(image)
        image_att = torch.ones(image_embed.size()[:-1], dtype=torch.long).to(self.device)
        text_embed = self.model.tokenizer(texts, padding='longest', truncation=True,
                                         max_length=40, return_tensors="pt").to(self.device)

        text_embed.input_ids[:, 0] = self.model.tokenizer.bos_token_id
        true_decoder_target = text_embed.input_ids.masked_fill(text_embed.input_ids == self.model.tokenizer.pad_token_id, -100)

        adv_texts = self.model.tokenizer(adv_captions, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(self.device)
        adv_texts.input_ids[:, 0] = self.model.tokenizer.bos_token_id
        adv_decoder_targets = adv_texts.input_ids.masked_fill(adv_texts.input_ids == self.model.tokenizer.pad_token_id, -100)

        adv_decoder_targets[:, :prompt_length] = -100
        true_decoder_target[:, :prompt_length] = -100

        adv_logits = self.model.text_decoder(adv_texts.input_ids,
                                             attention_mask=adv_texts.attention_mask,
                                             encoder_hidden_states=adv_embeds,
                                             encoder_attention_mask=adv_atts,
                                             labels=adv_decoder_targets,
                                             return_dict=True,
                                             ).logits

        true_logits = self.model.text_decoder(text_embed.input_ids,
                                              attention_mask=text_embed.attention_mask,
                                              encoder_hidden_states=image_embed,
                                              encoder_attention_mask=image_att,
                                              labels=true_decoder_target,
                                              return_dict=True,
                                              ).logits

        adv_logits, true_logits = adv_logits[:, :-1, :].contiguous(), true_logits[:, :-1, :].contiguous()

        padding_size = adv_logits.size(1) - true_logits.size(1)
        if padding_size > 0:
            padded_logits = F.pad(true_logits, (0, 0, 0, padding_size, 0, 0))
            padded_logits[:, -padding_size:, :] = -100
            loss_kl = criterion(adv_logits[:, :, 0].log_softmax(dim=-1), padded_logits[:, :, 0].softmax(dim=-1))
        elif padding_size < 0:
            padded_logits = F.pad(true_logits, (0, 0, padding_size, 0, 0, 0))
            loss_kl = criterion(adv_logits[:, :, 0].log_softmax(dim=-1), padded_logits[:, :, 0].softmax(dim=-1))
        else:
            loss_kl = criterion(adv_logits[:, :, 0].log_softmax(dim=-1), true_logits[:, :, 0].softmax(dim=-1))

        return loss_kl

    def _compute_kl_loss(self, caption_inputs, adv_embeds, ori_embed):

        criterion = torch.nn.KLDivLoss(reduction='batchmean')
        text_ids = self.model.tokenizer(caption_inputs, return_tensors="pt", padding=True, truncation=True).input_ids.to(self.device)
        text_embeds = self.model.text_decoder.bert.embeddings(input_ids=text_ids)
    
        if self.args.model in ['Blip2OPT', 'blip2_vicuna_instruct', 'XVLM']:
            image_proj = nn.Linear(1024, 768).to(self.device)
            adv_img_embeds = image_proj(adv_embeds)[:, 0, :]
            adv_embeds = adv_embeds[:, 0, :]
            text_embeds = text_embeds[:, 0, :]
            loss_kl_txt = criterion(adv_img_embeds.log_softmax(dim=-1), text_embeds.softmax(dim=-1))
        else:
            adv_embeds = adv_embeds[:, 0, :]
            text_embeds = text_embeds[:, 0, :]
            loss_kl_txt = criterion(adv_embeds.log_softmax(dim=-1), text_embeds.softmax(dim=-1))
    
        ori_embed = ori_embed[:, 0, :].detach()
        loss_kl = loss_kl_txt + criterion(adv_embeds.log_softmax(dim=-1), ori_embed.softmax(dim=-1))
    
        return loss_kl
