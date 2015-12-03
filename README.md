# NB_project

Below is example for PDF id = 15456.

Split PDF into a PDF file for each page (I was lazy and used an online service, but there should be scripts to do this automatically). These are placed into /pdfs/ into a folder called 15456.split. 

Run convert_extract.py on the code which runs pdf2htmlEX on the pdf: https://github.com/coolwanglu/pdf2htmlEX
This converts each page into an HTML page and injects javascript code into the page, including coords for the bounding box and scripts to extract the text and pass results to a server. The script is called extract2.js.

Move the resulting folder of HTML pages into /nbserver/nbapp/static/nbapp/. I have a few different extract.js files in that folder because some converted HTML files didn't line up so well. You can now view the pages on a browser and see the bounding boxes appear. The extracted text is printed to the console. I added not only the text of the box but also interesting CSS characteristics (color, x-position, size) to help me determine if the text is a heading, a figure caption, etc.

If you have the server running, then the script attempts to write the extracted text back to the postgres DB via a Django url whenever the page loads.

Then run load_pages.py while the server is running which loads each page which dumps data into the database.

I do further cleaning. analyze_postgres_text_\*.py attempts to go through the extracted text to determine which ones are confusing (from the users' post message), counts the number per PDF segment, and tries to do some cleaning.
analyze_sentences_\*.py attempts to translate the data into confusion annotations per *sentence* and not page fragment or line, and also produce each sentence in order they logically appear, grouping sections together, and extracting figure, table, and heading information.

No extra code here but I also use Stanford's POS tagger and parser to get parse trees and POS tags for each final cleaned sentence.

There are some scripts for some simple models that I tried. Features include unigram model, word vector model trained on Google corpus, word vector model trained on my PDF training corpus, number of different POS tags, height of parse tree, avg word len/sentence, sentence len, frequency of words in the sentence in the training corpus, first mention of each word, etc. So far none perform very well.
