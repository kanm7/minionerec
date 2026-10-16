import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('downstream_builder',Path(__file__).parents[1]/'data/build_industrial_downstream_inputs.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


class DownstreamTests(unittest.TestCase):
    def test_three_products_are_nine_ordered_sid_tokens(self):
        mapping={str(i):[f'<a_{i}>',f'<b_{i}>',f'<c_{i}>'] for i in range(3)}
        products,tokens=builder.sid_sequence(['0','1'],'2',mapping)
        self.assertEqual(products,['0','1','2'])
        self.assertEqual(tokens,mapping['0']+mapping['1']+mapping['2'])
        self.assertEqual(len(tokens),9)

    def test_missing_mapping_is_not_silently_skipped(self):
        with self.assertRaises(KeyError):builder.sid_sequence(['missing'],'target',{'target':['a','b','c']})

    def test_description_matches_longest_text_and_preserves_source(self):
        original=['','short','longer description']
        item=dict(title='Title',description=original)
        builder.normalize_description(item)
        self.assertEqual(item['description'],'longer description')
        self.assertEqual(item['source_description'],original)


if __name__=='__main__':unittest.main()
