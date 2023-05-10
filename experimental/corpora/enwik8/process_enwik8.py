
import os, sys
import xml.etree.ElementTree as ET
import re
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError: # Already removed
    pass
import preprocessing_utils as pu
from preprocessing_utils import ProcessCorpus


class ProcessEnwik8(ProcessCorpus):
    def __init__(self, filepath, outputname):
        super().__init__(
            corpus_type="single_file",
            filepath=filepath, 
            outputname=outputname)
        
    def process_article_text(self, t):
        t = t.lower()
        t = re.sub(r'&gt;', '>', t) # Decode URL encoded chars
        t = re.sub(r'&lt;', '<', t)
        t = re.sub(r'&amp;', '&', t)
        t = re.sub(r'&quot;', "", t)
        #t = re.sub(r'\(\[\[[^|]*language\|[^\)]*\)', "", t)  # remove translations to other language
        #t = re.sub(r'\[\[[^|]*language\|[^\)]*\]\]:', "", t)  # remove invisible language text

        t = re.sub(r'<ref[^<]*</ref>', "", t) # Remove references
        t = re.sub(r'<[^>]*>', "", t) # Remove xhtml tags
        t = re.sub(r'\[http:[^]]*', "[", t) # Remove normal URL, preserve visible text
        t = re.sub(r'\|thumb', "", t) # Remove images links, preserve captions
        t = re.sub(r'\|left', "", t)  
        t = re.sub(r'\|right', "", t)
        t = re.sub(r'\|\d+px', "", t)
        t = re.sub(r'\[\[image:[^\[\]]*\|', "", t)
        t = re.sub(r'\[\[category:([^|\]]*)[^]]*\]\]', '[[$1]]', t)  # show categories without markup
        t = re.sub(r'\[\[[a-z\-]*:[^\]]*\]\]', "", t)
        t = re.sub(r'\[\[[^\|\]]*\|', "[[", t)  # remove wiki url, preserve visible text
        t = re.sub(r'{{[^}]*}}', '', t)         # remove {{icons}} and {tables}
        t = re.sub(r'{[^}]*}', "", t)
        t = re.sub(r'\[',"", t)                # remove [ and ]
        t = re.sub(r'\]', "", t)
        t = re.sub("\n", " ", t)
        t = re.sub(r'\w*[#]\w*', "", t) # Remove any word containing #
        t = re.sub(r'\w*[\\]\w*', "", t) # Remove any word containing \
        t = re.sub(r'&[^;]*;',"", t)       # remove URL encoded chars

        # Generic preprocessing
        t = t.lower()
        t = t.replace("%", " percent ")
        t = t.replace("\t", " ")

        return t

    def custom_file_func(self, text):
        tree = ET.parse(os.path.join(Path(__file__).parent, "data/enwik8"))
        root = tree.getroot()
        print(root)

        texts = []
        count = 0

        for x in root.findall(
            '{http://www.mediawiki.org/xml/export-0.3/}page'
            ):
            tit = x.find(
                '{http://www.mediawiki.org/xml/export-0.3/}title'
                ).text
            print(tit)
            if "List" not in tit:
                for y in x.findall(
                    '{http://www.mediawiki.org/xml/export-0.3/}revision'
                    ):
                    article_text = y.find(
                        '{http://www.mediawiki.org/xml/export-0.3/}text'
                        ).text
                    if article_text is not None:
                        if (
                            "#REDIRECT" not in article_text
                            ) & (
                            "#redirect" not in article_text) & (
                            "#Redirect" not in article_text):
                            article_text = self.process_article_text(article_text)
                            if (
                                "#redirect" not in article_text
                                ) and (
                                    len(article_text) > 0
                                ) and (
                                    article_text != "moved to "
                                ):
                                texts.append(article_text)

        return texts


if __name__ == "__main__":

    ProcessEnwik8(filepath=os.path.join(Path(__file__).parent, "data/enwik8"),
                   outputname="enwik8")
