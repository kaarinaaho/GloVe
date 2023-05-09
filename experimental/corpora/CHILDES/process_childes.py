
import os, sys
import re
import shutil
# Aim: compile all CHILDES documents, each file [conversation] as a single line
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


class ProcessCHILDES(ProcessCorpus):
    def __init__(self, dirlist, outputname):
        super().__init__(
            corpus_type="directories",
            dirlist=dirlist, 
            outputname=outputname)

    def line_process(self, line):
        line = line.lower()
        add = line.replace("\n", " ")
        add = add.replace(
                            "xxx", ""
                            ).replace(
                                "yyy", ""
                                ).replace(
                                    "\t", " "
                                    ).replace(
                                        "www", ""
                                        ).replace(
                                            "zzz",""
                                        )
        add = re.sub(r'\[[^)]*\]', "", add)
        add = re.sub(r'\&[^)]* ', "", add)

            
        # Remove speaker name from beginning of line
        if ":" in add:
            add = add.split(":", 1)[1]
        
        add = add.replace("q ", " ")
        add = self._remove_numbers(add)

        return add

    def custom_file_func(self, text):
        """Take file --> string."""
        doc = []
        prev_line_permitted=False
        for line in text:
            if (
                line[0] not in ["@", "%", " ", "\t"]
            ) & (
                "CHI" not in line[:5]
            ) & (":" in line) & (len(line) > 0):
                
                add = self.line_process(line)
                # If resultant line contains content, add it to document
                if len(add.replace(" ", "")) > 0:
                    doc.extend(add)

                    prev_line_permitted = True
                else:
                    prev_line_permitted = False

            elif (line[0] == "\t") & (prev_line_permitted):

                add = self.line_process(line)
                # If resultant line contains content, add it to document
                if len(add.replace(" ", "")) > 0:
                    doc.extend(add)

                    prev_line_permitted = True

                else:
                    prev_line_permitted = False

            else:
                prev_line_permitted = False

        # Compile document into single string
        doc = "".join(doc)

        # Return document string
        return doc
        

if __name__ == "__main__":

    ProcessCHILDES(dirlist=[os.path.join(
                                Path(__file__).parent, "data/over24months"),
                            os.path.join(
                                Path(__file__).parent, "data/under24months")],
                   outputname="childes_all")
