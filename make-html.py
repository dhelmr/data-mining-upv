#!/bin/python3.7

"""
This script takes the wordclouds and original tweets of a clustering results
and creates a little HTML presentation that allows to scroll through the clusters.
The html page is served on localhost, the port can be specified.
"""

TEMPLATE = """
<html>
    <head>
    <title>Clustering results</title>
    <style>
* {box-sizing: border-box}
body {font-family: Verdana, sans-serif; margin:0}
.mySlides {display: none}
img {vertical-align: middle;}

/* Slideshow container */
.slideshow-container {
  max-width: 40%;
  position: relative;
  margin: 1em auto auto;
}

/* Next & previous buttons */
.prev, .next {
  cursor: pointer;
  position: absolute;
  top: 50%;
  width: auto;
  padding: 16px;
  margin-top: -22px;
  color: white;
  font-weight: bold;
  font-size: 18px;
  transition: 0.6s ease;
  border-radius: 0 3px 3px 0;
  user-select: none;
}

/* Position the "next button" to the right */
.next {
  right: 0;
  border-radius: 3px 0 0 3px;
}

/* On hover, add a black background color with a little bit see-through */
.prev:hover, .next:hover {
  background-color: rgba(0,0,0,0.8);
}

    body { 
      margin: auto;
      max-width: 1200px;
      display: block;
    }

#tweetbox {
width: 100%;

height: 300px;

background-color: #f1f1f1;

font-size: 1.5em;

resize: none;

border: 1px solid #BFBFBF;
box-shadow: 5px 5px 5px #aaaaaa;
}
	
	


/* Caption text */
.text {
  color: #f2f2f2;
  font-size: 15px;
  padding: 8px 12px;
  position: absolute;
  bottom: 8px;
  width: 100%;
  text-align: center;
}

/* Number text (1/3 etc) */
.numbertext {
  color: #f2f2f2;
  font-size: 12px;
  padding: 8px 12px;
  position: absolute;
  top: 0;
}

/* The dots/bullets/indicators */
.dot {
  cursor: pointer;
  height: 15px;
  width: 15px;
  margin: 0 2px;
  background-color: #bbb;
  border-radius: 50%;
  display: inline-block;
  transition: background-color 0.6s ease;
}

.active, .dot:hover {
  background-color: #717171;
}

/* Fading animation */
.fade {
  -webkit-animation-name: fade;
  -webkit-animation-duration: 1.5s;
  animation-name: fade;
  animation-duration: 1.5s;
}

@-webkit-keyframes fade {
  from {opacity: .4} 
  to {opacity: 1}
}

@keyframes fade {
  from {opacity: .4} 
  to {opacity: 1}
}

/* On smaller screens, decrease text size */
@media only screen and (max-width: 300px) {
  .prev, .next,.text {font-size: 11px}
}
    </style>
    </head>
    <body>
        <div class="slideshow-container">
        $slideshow_elements
        <a class="prev" onclick="plusSlides(-1)">&#10094;</a>
        <a class="next" onclick="plusSlides(1)">&#10095;</a>
        </div>
        <br>
        <div id="caption"></div>
        <textarea id="tweetbox"> </textarea>
           <script lang="text/javascript">
        let tweetSamples = [
            $tweet_samples
        ]
        tweetbox = document.getElementById("tweetbox")
    
        var slideIndex = 1;
        showSlides(slideIndex);

        // Next/previous controls
        function plusSlides(n) {
        showSlides(slideIndex += n);
        }

        // Thumbnail image controls
        function currentSlide(n) {
        showSlides(slideIndex = n);
        }

        function showSlides(n) {
            var i;
            var slides = document.getElementsByClassName("mySlides");
            console.log(slides)
            if (n > slides.length) {slideIndex = 1}
            if (n < 1) {slideIndex = slides.length}
            for (i = 0; i < slides.length; i++) {
                slides[i].style.display = "none";
            }
            slides[slideIndex-1].style.display = "block";
            set_samples(n-1);
        } 
        function set_samples(n) {
            path = tweetSamples[n].path;
            var client = new XMLHttpRequest();
            client.open('GET', path);
            client.onreadystatechange = function() {
              text = client.responseText
              document.getElementById("tweetbox").value = text
            }
            client.send();
            document.getElementById( "caption").innerHTML = "Cluster " + (n+1)+": "+  tweetSamples[n].length + " tweets"
            
        }
        function load_text(path) {
            
        }
    </script>
    </body>
</html>
"""

SLIDE_SHOW_TEMPLATE = """
    <div class="mySlides fade $active">
    <div class="numbertext">$index / $total</div>
    <img src="$image_path" style="width:100%">
    <div class="text">$caption</div>
    </div>
"""
from os import listdir, chdir
from os.path import isfile, join
import tempfile
from shutil import copy
import http.server
import socketserver
import argparse

tmp_dir = tempfile.mkdtemp()

def make_html(slideshow_html, tweet_samples):
    return TEMPLATE.replace("$tweet_samples", tweet_samples).replace("$slideshow_elements", slideshow_html)

def make_slideshow_element(index, total, img_path, caption, isActive=False) -> str:
    active = ""
    if isActive == True:
        active = "active"
    return SLIDE_SHOW_TEMPLATE.replace("$index", str(index+1)).replace("$total", str(total)).replace("$image_path", img_path).replace("$caption", caption).replace("$active", active)
    

def ls_files(directory, ending):
    return [f for f in listdir(directory) if isfile(join(directory, f)) and str(f).endswith(ending)]

def make_sample_js(file, index, length):
    return "{'path': '"+file+"', 'length': '"+str(length)+"'}"

def main(wordcloud_dir, text_dir, port):
    image_files = ls_files(wordcloud_dir, ".png")
    slideshow_html = ""
    sample_js = []
    for i in range(len(image_files)):
        active = False
        if i == 0:
            active = True
        f = f"{i}.txt.png"
        copy(join(wordcloud_dir, f), tmp_dir)
        slideshow_html += make_slideshow_element(i, len(image_files), f, "", isActive=active)
        textfile = f"{i}.txt"
        length = num_lines = sum(1 for line in open(join(text_dir, textfile)))
        copy(join(text_dir, textfile), tmp_dir)
        sample_js.append( make_sample_js(textfile, i, length) + "\n")
        

    html = make_html(slideshow_html, ",".join(sample_js))
    f = open(join(tmp_dir, "index.html"), "w")
    f.write(html)
    f.close()

    chdir(tmp_dir)
    Handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("", port), Handler)
    print("serving at port", port)
    httpd.serve_forever()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates a html presentation of the clustering results and serves it at localhost')
    parser.add_argument(
        '--texts', dest='text_dir', help='Folder that contains the original tweets grouped by clusters', required=True)
    parser.add_argument('--wordclouds', dest='wordcloud_dir', required=True,
                        help='Folder that contains the wordcloud images')
    parser.add_argument('--port', dest='port', default=8123, type=int,
                    help='port where the server will run')
    args = parser.parse_args()
    
    main(args.wordcloud_dir, args.text_dir, args.port)
