# __version__ = '0.1.7'

try:
    # noinspection PyPackageRequirements
    import torch
except ImportError:
    raise ImportError('Using t5mem requires torch. Please refer to https://pytorch.org/get-started/locally/ '
                      'to install the correct version for your setup')

from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM

from .configuration_t5mem import T5MemConfig


print("from .configuration_t5mem import T5MemConfig  done! ")
# noinspection PyUnresolvedReferences
from .modeling_t5mem import T5MemModel, T5MemForConditionalGeneration
print("from .modeling_t5mem import T5MemModel, T5MemForConditionalGeneration  done")
from .tokenization_t5mem import T5MemTokenizer
print("from .tokenization_t5mem import T5MemTokenizer done")
from .tokenization_t5mem_fast import T5MemTokenizerFast
print("from .tokenization_t5mem_fast import T5MemTokenizerFast  done!")


AutoConfig.register('t5mem', T5MemConfig)
AutoModel.register(T5MemConfig, T5MemModel)
AutoModelForSeq2SeqLM.register(T5MemConfig, T5MemForConditionalGeneration)
# AutoTokenizer.register(T5MemConfig, slow_tokenizer_class=T5MemTokenizer, fast_tokenizer_class=T5MemTokenizerFast)


AutoConfig.register('tau/sled', SledConfig)
AutoModel.register(SledConfig, SledModel)
AutoModelForSeq2SeqLM.register(SledConfig, SledForConditionalGeneration)
AutoTokenizer.register(SledConfig, slow_tokenizer_class=SledTokenizer, fast_tokenizer_class=SledTokenizerFast)
