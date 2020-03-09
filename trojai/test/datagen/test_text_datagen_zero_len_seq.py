import unittest

from trojai.datagen.text_entity import GenericTextEntity

"""
We notice that if a text sequence, after delimiting, has sequences of length 0, then
TorchText doesn't like it.  Example of a file which works is in:
files/9139_8.txt.works
Example of a file which is broken is:
files/9139.8.txt.broken

This seems to be happening because of the way get_text() is implemented in text_entity.py
Here, we join all the delimited data by spaces.  However, suppose we have a sentence where 
it looks like:
abc 123 ...
This will get reconstructed as:
abc 123 . . .
After tokenization, this yields problematic sequences


The stacktrace is here:
  File "text_classification.py", line 215, in <module>
    model_generator.run()
  File "/home/karrak1/trojai/trojai/trojai/modelgen/model_generator.py", line 46, in run
    runner.run()
  File "/home/karrak1/trojai/trojai/trojai/modelgen/runner.py", line 110, in run
    self.progress_bar_disable)
  File "/home/karrak1/trojai/trojai/trojai/modelgen/lstm_optimizer.py", line 258, in train
    progress_bar_disable=progress_bar_disable)
  File "/home/karrak1/trojai/trojai/trojai/modelgen/lstm_optimizer.py", line 325, in train_epoch
    predictions = model(text, text_lengths).squeeze(1)
  File "/home/karrak1/virtual_environments/trojai/lib/python3.5/site-packages/torch/nn/modules/module.py", line 493, in __call__
    result = self.forward(*input, **kwargs)
  File "/home/karrak1/trojai/trojai_private/trojai_private/modelgen/architectures/text_architectures.py", line 45, in forward
    packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
  File "/home/karrak1/virtual_environments/trojai/lib/python3.5/site-packages/torch/nn/utils/rnn.py", line 268, in pack_padded_sequence
    torch._C._VariableFunctions._pack_padded_sequence(input, lengths, batch_first)
RuntimeError: Length of all samples has to be greater than 0, but found an element in 'lengths' that is <= 0
"""


class TestTextDatagenZeroLenSeq(unittest.TestCase):
    def test_from_file(self):
        files_dir = 'files/zero_len_sequences/'
        working_file_data = "The movie is okay, it has it's moments, the music scenes are the best of all! The " \
                            "soundtrack is a true classic. It's a perfect album, it starts out with Let's Go Crazy" \
                            "(appropriate for the beginning as it's a great party song and very up-tempo), Take Me With " \
                            "U(a fun pop song...), The Beautiful Ones(a cheerful ballad, probably the closest thing to " \
                            "R&B on this whole album), Computer Blue(a somewhat angry anthem towards Appolonia), " \
                            "Darling Nikki(one of the funniest songs ever, it very vaguely makes fun of Appolonia), " \
                            "When Doves Cry(the climax to this masterpiece), I Would Die 4 U, Baby I'm A Star, and, " \
                            "of course, Purple Rain(a true classic, a very appropriate ending for this classic album) " \
                            "The movie and the album are both very good. I highly recommend them!"
        broken_file_data = "The movie is okay, it has it's moments, the music scenes are the best of all! The " \
                           "soundtrack is a true classic. It's a perfect album, it starts out with Let's Go Crazy(" \
                           "appropriate for the beginning as it's a great party song and very up-tempo), Take Me " \
                           "With U(a fun pop song. . . ), The Beautiful Ones(a cheerful ballad, probably the closest " \
                           "thing to R&B on this whole album), Computer Blue(a somewhat angry anthem towards Appolonia), " \
                           "Darling Nikki(one of the funniest songs ever, it very vaguely makes fun of Appolonia), " \
                           "When Doves Cry(the climax to this masterpiece), I Would Die 4 U, Baby I'm A Star, and, of " \
                           "course, Purple Rain(a true classic, a very appropriate ending for this classic album) " \
                           "The movie and the album are both very good. I highly recommend them!"

        text_data_entity = GenericTextEntity(working_file_data)
        print(text_data_entity.get_delimiters())
        print(text_data_entity.get_data())
        reconstructed_data = text_data_entity.get_text()
        print(working_file_data)
        print(reconstructed_data)

    def test_raw_string(self):
        str1 = "hello 123. abc ..."
        e = GenericTextEntity(str1)
        print(e.get_data())
        print(e.get_delimiters())
        print()
        print(str1)
        print(e.get_text())


if __name__ == '__main__':
    unittest.main()
