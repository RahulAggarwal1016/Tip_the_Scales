//express is used to create a server
//multer handles the files
//cors enables cross-origin request to the server
//Nodemon is used to start the server

var express = require("express");
var app = express();
var multer = require("multer");
var cors = require("cors");
const { exec, execSync } = require("child_process");
app.use(cors());

//create a multer instance and set the destination folder
var storage = multer.diskStorage({
	destination: function (req, file, cb) {
		cb(null, "../html/python/"); //src
	},
	filename: function (req, file, cb) {
		cb(null, "audio_input.wav"); //file.originalname
	},
});

//create an upload instance and recieve files

//below is the post route to upload the file
app.post("/upload", multer({ storage: storage }).array("file"), function (
	req,
	res
) {
	const process = execSync("python3.7 ../html/python/convert-to-jpg.py");
	return res.status(200).send(req.file);
});

//Make the server listen on port 8000
app.listen(8000, function () {
	console.log("App running on port 8000");
});

//How to run the full application
//1. Make sure to cd into the "scales" directory
//2. Type in the terminal "npm start scales"
//3. Open another terminal (may have to cd into "scales" again) and type "npx nodemon server.js"

// pkill -f node --> Kill the server (server won't run again unless killed)
// quit() --> Exit the react web app (closing the tab won't close it)
