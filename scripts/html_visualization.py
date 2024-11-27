import glob
import os, sys
import shutil
import math
import numpy as np
from scripts.util import mkdir, read_txt, write_txt

COLORS = [
    "black",
    "red",
    "yellow",
    "blue",
    "green",
    "gray",
    "maroon",
    "purple",
    "fushsia",
    "lime",
    "olive",
    "silver",
    "navy",
    "blue",
    "teal",
    "aqua",
]


class HtmlGenerator(object):
    def __init__(
        self,
        input_folder,
        output_folder,
        subfolders,
        color_labels = None,
        host_address=None,
        num_user=1,
        num_column=10,
    ):
        self.input_folder = input_folder
        self.color_labels = ["undefined"]
        if color_labels is not None:
            self.color_labels = color_labels

        # Get the subfolders in the given folder directory
        self.subfolders = subfolders

        # files for storage
        self.output_folder = output_folder
        self.saved_folder = os.path.join(output_folder, "saved_%d")
        self.test_folder = os.path.join(output_folder, "test_%d")
        self.test_file = os.path.join(self.test_folder, "test_%d.html")
        self.js_folder = os.path.join(output_folder, "js")

        # files for display
        self.use_php = host_address is not None
        self.host_address = (
            os.path.join(host_address, "test_%d")
            if host_address is not None
            else self.test_folder
        )
        self.num_user = num_user
        self.num_column = num_column

        # will be computed by the input
        self.num_pages = len(self.subfolders)
        self.image_counts = []
        self.num_per_user = []
        self.num_last_user = []

        self.setup_www()
        self.setup_users()

    def setup_www(self):
        if self.use_php:
            php_file = os.path.join(self.output_folder, "save_ans.php")
            if not os.path.exists(php_file):
                shutil.copy(os.path.join("www", "save_ans.php"), php_file)

        mkdir(self.js_folder)
        for fn in ["jquery-1.7.1.min.js", "util.js"]:
            js_file = os.path.join(self.js_folder, fn)
            if not os.path.exists(js_file):
                shutil.copy(
                    os.path.join("www", "js", fn),
                    js_file,
                )

    def setup_users(self):
        for i in range(self.num_user):
            # folder of saved results
            folder_save = self.saved_folder % i
            mkdir(folder_save)
            # need to be writeable from browser
            os.chmod(folder_save, 0o777)
            # folder of test htmls
            folder_test = self.test_folder % i
            mkdir(folder_test)

    def create_html(self, all_image_paths, all_image_labels):

        # Loop through each subfolder in the input_folder and calculate image counts
        for subfolder_image_paths in all_image_paths:
            self.image_counts.append(len(subfolder_image_paths))  # Store the number of images in each subfolder

        # num_per_user is now a list, each value corresponds to the evenly divided number of images for each subfolder
        self.num_per_user = [int(math.ceil(image_count / self.num_user)) for image_count in self.image_counts]

        # Calculate the number of remaining images for the last user in each subfolder
        self.num_last_user = [image_count - (self.num_user - 1) * num_per for image_count, num_per in
                              zip(self.image_counts, self.num_per_user)]

        self.create_html_index()
        self.create_html_proofreading(
            all_image_paths, all_image_labels, self.use_php
        )

    def create_html_index(self):
        # if on server, index.html can automatically present the tasks to work on
        # if on client, js can't check file existence
        if self.use_php:
            save_pref = "saved_%d/s_" if self.use_php else "saved_%d_"
            for i in range(self.num_user):
                self.num_pages = len(self.subfolders)
                out = """
                <html>
                    <head>
                        <script src="js/jquery-1.7.1.min.js"></script>
                        <script src="js/util.js"></script>
                    </head>
                    <body>
                        <h1 id="tt"></h1>
                        <script>
                        var st=parseInt(getUrlParam('st'));if(isNaN(st)){st=0;}
                            var num = %d;
                            function check_file(fileID){
                                if(fileID >= num){
                                    $('#tt').text("Congratulations! Mission Completed!!"); 
                                }else{
                                    $.get('saved_%d/%s'+fileID+'.txt').done(function(){
                                            check_file(fileID+1)
                                        }).fail(function(){
                                            window.location = 'test_%d/test_'+fileID+'.html'
                                        })
                                    }
                            }
                            check_file(st)
                        </script>
                    </body>
                </html>
                """ % (
                    self.num_pages,
                    i,
                    save_pref % i,
                    i,
                )
                write_txt(os.path.join(self.output_folder, f"index_{i}.html"), out)

    def create_html_proofreading(
        self, all_image_paths, all_image_labels=None, use_php=False
    ):
        if all_image_labels is None:
            all_image_labels = [np.zeros(len(paths), dtype=int) for paths in all_image_paths]

        colors = COLORS[: len(self.color_labels)]
        color_js = '["' + '","'.join(colors) + '"]'

        for uid in range(self.num_user):
            for fid in range(self.num_pages):
                image_paths = all_image_paths[fid]
                image_labels = all_image_labels[fid]
                num_images = self.num_per_user[fid] if uid != self.num_user - 1 else self.num_last_user[fid]

                # Define start and end indices for the current user's images
                id_start = self.num_per_user[fid] * uid
                id_end = id_start + num_images

                output_file = self.test_file % (uid, fid)
                out = "<html>"
                out += """
                <style>
                .im0{position:absolute;top:0px;left:0px;height:100%;}
                .im1{position:relative;top:-75px;left:-75px;opacity:0.7;}
                .im2{position:absolute;top:-75px;left:-75px;opacity:0.3;}
                .crop{position:relative;left:0;top:0;overflow:hidden;width:900px;height:270px;}
                </style>
                """
                # color scheme
                out += f"""
                <script src="../js/jquery-1.7.1.min.js"></script>
                <h1> User {uid}, Category: {self.color_labels[fid+1]}, Page {fid+1}/{self.num_pages}</h1>
                <h3> color coding</h3>
                <ul>"""
                for i, color_name in enumerate(self.color_labels):
                    out += f"<li>{colors[i]}: {color_name}</li>\n"
                out += "</ul>\n"
                # display
                out += "<table cellpadding=8 border=2>\n"

                for i in range(id_start, id_end):
                    i_rel = i - id_start
                    if (i_rel % self.num_column) == 0:
                        out += "\t <tr>\n"
                    out += (
                        f'\t\t<td id="{os.path.splitext(os.path.basename(image_paths[i][0]))[0]}" class="cc"'
                        f' bgcolor="{colors[image_labels[i]]}">\n'
                    )
                    for image_path in image_paths[i]:
                        if isinstance(image_path, list):
                            # overlay display
                            out += f"""
                                <div class="crop" >
                                    <img class="im1" src="{image_path[0]}">
                                    <img class="im2" src="{image_path[1]}">
                                </div>
                                """
                        else:
                            out += f"""
                                <div class="crop">
                                    <img class="im0" src="{image_path}">
                                </div>
                                """
                    out += "</td>\n"
                    if (i_rel + 1) % self.num_column == 0:
                        out += "</tr>\n"
                out += "</table>\n"
                # submission
                out += """<table border=2>
                    <tr>
                        <td>
                            <button id="sub" style="width:100%;height=40">Done</button>
                        </td>
                        <td>
                            <button id="next" style="width:100%;height=40">Next</button>
                        </td>
                    </tr>
                    </table>
                    """
                if use_php:
                    out += f"""
                    <form id="mturk_form" method="POST" style="display:none">
                        <input id="task" name="task" value="saved_{uid}/s_">
                        <input id="ans" name="ans">
                        <input id="fileId" name="fileId" value="{fid}">
                    </form>
                    """

                # js
                out += f"""
                    <script>
                        TOTAL_I={id_end - id_start};
                        colors={color_js};
                    """
                out += """
                function get_answer() {
                    var out='';
                    $(".cc").each(function(index) {  // Loop through all elements with the class "cc"
                        var vesicleId = this.id;  // Get the vesicle ID from the id attribute
                        var cc = this.getAttribute("bgcolor");  // Get the background color
                        var colorIndex = Math.max(0, colors.indexOf(cc));  // Get the color index
                        out += `(${vesicleId}:${colorIndex})`;  // Format as (vesicle id:color id)
                        if (index < $(".cc").length - 1) {
                            out += ',';  // Separate pairs with commas
                        }
                    });
                    return out;
                }
                $(".cc").click(function(){
                    var currentColor = $(this)[0].getAttribute("bgcolor");
                    var nextColor = colors[(colors.indexOf(currentColor) + 1) % colors.length];
                     $(this)[0].setAttribute("bgcolor", nextColor);
                });
                """
                out += """
                $("#next").click(function(){
                    window.location = '../test_%d/test_%d.html'
                });
                """ % (
                    uid,
                    fid + 1,
                )
                if use_php:
                    # write files on the server
                    out += """
                    $("#sub").click(function(){
                        ans_out=get_answer();
                        document.getElementById("ans").value = ans_out;
                        tmp = $.post("../save_ans.php", $("#mturk_form").serialize(),function(data) {
                            window.location='%s?st=%d';                    
                        });
                    });
                    """ % (
                        self.host_address % uid,
                        fid,
                    )
                else:
                    # write files with browser
                    out += """
                    $("#sub").click(function(){                        
                        const link = document.createElement("a");
                        const content = get_answer();
                        const file = new Blob([content], { type: 'text/plain' });
                        link.href = URL.createObjectURL(file);
                        link.download = "saved_%d_%d.txt";
                        link.click();
                        URL.revokeObjectURL(link.href);
                    });
                    """ % (
                        uid,
                        fid,
                    )

                out += "</script>\n</html>"
                write_txt(output_file, out)

    def create_html_summary(self, labels, image_paths, output_file):
        # image_paths: list of image lists
        out = (
            "<html>"
            + """
            <style>
            .crop{position:relative;left:0;top:0;overflow:hidden;width:150px;height:150px;}
            </style>
            """
            + '<script src="../js/jquery-1.7.1.min.js"></script>'
        )
        for label, image_path in zip(labels, image_paths):
            out += f"<h1>Visualize {label} (#={len(image_path)})</h1>\n"
            out += "<table>\n"
            for index, path_list in enumerate(image_path):
                if index % self.num_column == 0:
                    out += "<tr>\n"
                out += '<td><div class="crop">'
                for path in path_list:
                    out += (
                        "<img"
                        ' style="position:relative;top:-75px;left:-75px;opacity:0.5;"'
                        f'  src="{path}">'
                    )
                out += "</div></td>\n"
                if (index + 1) % self.num_column == 0:
                    out += "</tr>\n"
            out += "</table>\n"
        out += "</html>"
        write_txt(output_file, out)
