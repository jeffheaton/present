
# Download Wikipedia Dump Data

This part only needs to be run once, at the beginning to download Wikipedia to somewhere on your system.  Note, this part will take a long time!  It will transfer much data!

# Process Wikipedia Data

This can be run after the previous section has downloaded Wikipedia data.  This example creates 3 files from extracted Wikipedia data:

* article.csv - A listing of all articles on Wikipedia, with their Wikipedia ID.
* redirect.csv - A listing of all redirects of articles on Wikipedia. e.g. USA to United_States
* template.csv - A listing of all templates on Wikipedia.  These are the "types" of articles.

# Running

nohup python example_cats_n_links.py >/home/log.txt &