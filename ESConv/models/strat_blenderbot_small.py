# coding=utf-8
# copied from bart

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import BaseModel

try:
    from transformers.generation_utils import top_k_top_p_filtering
except ImportError:
    from transformers import top_k_top_p_filtering
from transformers.models.blenderbot_small import (BlenderbotSmallConfig, BlenderbotSmallForConditionalGeneration, )
from transformers.modeling_outputs import (BaseModelOutput, Seq2SeqModelOutput, Seq2SeqLMOutput, )
from .PARAMS import SAMPLE, TEMPERATURE


class Model(BaseModel, BlenderbotSmallForConditionalGeneration):
    def __init__(self, config: BlenderbotSmallConfig):
        super().__init__(config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_input_idss=None,
            encoder_outputs=None,
            past_key_values=None,
            labels=None,
            use_cache=None,
            return_dict=None,
            validation=False,
            align=False,
            labelss=None,

            **kwargs
    ):
        assert self.toker is not None
        training = False if align else self.training  

        encoded_info = kwargs
        assert (training or validation) == (labels is not None)
        if validation:  
            labels[:, 0] = -100
            labelss[:, 0] = -100

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if not self.training and not validation:  # inference
            use_cache = True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict



        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict,
        )

        


        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias


        if validation:
            lm_logits = lm_logits[..., :self.toker.vocab_size].contiguous()

        masked_lm_loss = None
        if labels is not None:
            loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1), reduction='none')
            loss = loss.view(labels.size(0), labels.size(1))
            label_size = torch.sum(labels.ne(-100), dim=1).type_as(loss)
            masked_lm_loss = torch.sum(loss) / torch.sum(label_size)
            ppl_value = torch.exp(torch.mean(torch.sum(loss, dim=1).float() / label_size.float()))

           


        # if not self.training and not validation:  # inference
        if not training and not validation: 
            if not return_dict:
                output = (lm_logits,) + outputs[1:]
                return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

            return Seq2SeqLMOutput(
                loss=masked_lm_loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )

        # elif self.training:  # training
        elif training:  # modi
            assert not validation
            # res = {'all': masked_lm_loss, 'ppl': ppl_value,"all1" :  masked_lmp_loss}
            res = {'all': masked_lm_loss, 'ppl': ppl_value}
            return res

        else:  # validation
            assert not self.training
            return loss, label_size
        



class Model_situation(BaseModel, BlenderbotSmallForConditionalGeneration):
    def __init__(self, config: BlenderbotSmallConfig):
        super().__init__(config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_input_idss=None,
            encoder_outputs=None,
            past_key_values=None,
            labels=None,
            use_cache=None,
            return_dict=None,
            validation=False,
            align=False,
            labelss=None,

            **kwargs
    ):
        assert self.toker is not None
        training = False if align else self.training  

        encoded_info = kwargs
        assert (training or validation) == (labels is not None)
        if validation:  
            labels[:, 0] = -100
            labelss[:, 0] = -100

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if not self.training and not validation:  # inference
            use_cache = True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict,
        )
# situation
        # outputs = self.model(
        #             input_ids,
        #             attention_mask=attention_mask,
        #             decoder_input_ids=decoder_input_idss,
        #             encoder_outputs=encoder_outputs,
        #             past_key_values=past_key_values,
        #             use_cache=use_cache,
        #             return_dict=return_dict,
        #         )

        #         # print(inputp_ids)
        #         # print(attentionp_mask)

        #         # outputsp = self.model(
        #         #     inputp_ids,
        #         #     attention_mask=attentionp_mask,
        #         #     decoder_input_ids=decoder_input_ids,
        #         #     encoder_outputs=None,
        #         #     past_key_values=None,
        #         #     use_cache=use_cache,
        #         #     return_dict=return_dict,
        #         # )


        # lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        # # lm_for_prediction_logits = self.lm_head(outputsp[0]) + self.final_logits_bias

        # # lmp_logits = self.lm_head(outputsp[0]) + self.final_logits_bias

        # if validation:
        #     lm_logits = lm_logits[..., :self.toker.vocab_size].contiguous()

        # masked_lm_loss = None
        # if labelss is not None:
        #     loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), labelss.view(-1), reduction='none')
        #     loss = loss.view(labelss.size(0), labelss.size(1))
        #     label_size = torch.sum(labelss.ne(-100), dim=1).type_as(loss)
        #     masked_lm_loss = torch.sum(loss) / torch.sum(label_size)
        #     ppl_value = torch.exp(torch.mean(torch.sum(loss, dim=1).float() / label_size.float()))

        #     # lossp = F.cross_entropy(lmp_logits.view(-1, lmp_logits.size(-1)), labels.view(-1), reduction='none')
        #     # lossp = lossp.view(labels.size(0), labels.size(1))
        #     # masked_lmp_loss = torch.sum(lossp) / torch.sum(label_size)
        
        # # if labels_for_prediction is not None:
        # #     loss1 = F.cross_entropy(lm_for_prediction_logits.view(-1, lm_for_prediction_logits.size(-1)), labels_for_prediction.view(-1), reduction='none')
        # #     loss1 = loss1.view(labels_for_prediction.size(0), labels_for_prediction.size(1))
        # #     label1_size = torch.sum(labels_for_prediction.ne(-100), dim=1).type_as(loss1)
        # #     masked_lm_loss1 = torch.sum(loss1) / torch.sum(label1_size)


        outputss = self.model(
            None,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_idss,
            encoder_outputs=outputs.encoder_last_hidden_state,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict,
        )

        


        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        lm_logitss = self.lm_head(outputss[0]) + self.final_logits_bias

        if validation:
            lm_logits = lm_logits[..., :self.toker.vocab_size].contiguous()

        masked_lm_loss = None
        masked_lm_losss = None

        if labels is not None:
            loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1), reduction='none')
            loss = loss.view(labels.size(0), labels.size(1))
            label_size = torch.sum(labels.ne(-100), dim=1).type_as(loss)
            masked_lm_loss = torch.sum(loss) / torch.sum(label_size)
            ppl_value = torch.exp(torch.mean(torch.sum(loss, dim=1).float() / label_size.float()))


        if labels is not None:
            losss = F.cross_entropy(lm_logitss.view(-1, lm_logitss.size(-1)), labelss.view(-1), reduction='none')
            losss = losss.view(labelss.size(0), labelss.size(1))
            label_sizes = torch.sum(labelss.ne(-100), dim=1).type_as(losss)
            masked_lm_losss = torch.sum(losss) / torch.sum(label_sizes)
            

           


        # if not self.training and not validation:  # inference
        if not training and not validation:  
            if not return_dict:
                output = (lm_logits,) + outputs[1:]
                return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

            return Seq2SeqLMOutput(
                loss=masked_lm_loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )

        # elif self.training:  # training
        elif training:  
            assert not validation

            masked_lm_loss = masked_lm_loss + 0.1*masked_lm_losss
            # res = {'all': masked_lm_loss, 'ppl': ppl_value,"all1" :  masked_lmp_loss}
            res = {'all': masked_lm_loss, 'ppl': ppl_value}
            return res

        else:  # validation
            assert not self.training
            return loss, label_size








    def predict_strategy(self, logits, encoded_info):
        assert not self.training
        strat_id = encoded_info.get('strat_id', None)
        logits = logits[:, 0, -8:]

        if strat_id is not None:
            pred = strat_id
        else:
            if SAMPLE:
                filtered_logits = top_k_top_p_filtering(logits / TEMPERATURE, top_p=0.9)
                pred = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).squeeze(-1)
            else:
                pred = torch.argmax(logits, dim=-1)

        pred_top1 = torch.topk(logits, k=1, dim=-1)[1]
        pred_top3 = torch.topk(logits, k=3, dim=-1)[1]

        encoded_info.update({
            'pred_strat_id': pred,
            'pred_strat_id_top1': pred_top1,
            'pred_strat_id_top3': pred_top3,
            'pred_strat_id_dist': F.softmax(logits, dim=-1)
        })

    @torch.no_grad()
    def generate(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            return_dict=None,
            input_for_prediction_ids = None,
            decoder_input_for_prediction_ids = None,
            labels_for_prediction = None,
            **kwargs
    ):
        assert not self.training
        assert self.toker is not None

        encoded_info = kwargs
        if decoder_input_ids is None:
            decoder_input_ids = torch.ones_like(input_ids[:, :1])
        assert decoder_input_ids.size(1) == 1
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )

        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(decoder_outputs.last_hidden_state) + self.final_logits_bias
        self.predict_strategy(lm_logits, encoded_info)

        decoder_input_ids = torch.cat(
            [decoder_input_ids, encoded_info['pred_strat_id'][..., None] + len(self.toker) - 8], dim=-1)

        assert 'max_length' in kwargs
        kwargs['max_length'] = kwargs['max_length'] + decoder_input_ids.size(1)
        kwargs['use_cache'] = True

        if len(self.toker) > self.toker.vocab_size:
            bad_words_ids = [[i] for i in range(self.toker.vocab_size, len(self.toker))]
            kwargs['bad_words_ids'] = bad_words_ids

        generations = super().generate(
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            **kwargs
        )
        return encoded_info, generations[:, decoder_input_ids.size(1):]
