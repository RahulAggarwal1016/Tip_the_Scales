//App.js is the main react file that allows the webapp to run
import React, { Component } from "react";
import "./App.css";
import axios from "axios";
import notes from "./frontendImages/music-notes.png";
import spectroImage from "./frontendImages/image.jpg";

// var url = "https://teachablemachine.withgoogle.com/models/un_XbaxjZ/";
var statement = "";

//the App class is the main component that renders to the screen
class App extends Component {
	constructor(props) {
		super(props);
		this.state = {
			selectedFile: null, //selected file state is set to null initially
			button_title: "SHOW SCALE SPECTROGRAM",
		};
	}
	render() {
		//Below is the react web app's jsx (what's displayed upon npm start)
		return (
			<div className="App">
				<head>
					<meta charset="UTF-8" />
					<title>Tip: The Scales</title>
				</head>

				<body>
					<div className="grey-rectangle">
						<h1 id="title" className="body-text">
							Tip: The Scales
						</h1>

						<div className="img-container">
							<img id="title-separation" src={notes}></img>
						</div>

						<p className="body-text"></p>
						<p className="body-text">
							The four most common types of scales used in western music are:{" "}
						</p>
						<p className="body-text unique2">
							<a href="https://en.wikipedia.org/wiki/Major_scale">
								<u>Major</u>
							</a>
							,&nbsp;
							<a
								className="body-text"
								href="https://en.wikipedia.org/wiki/Minor_scale#Natural_minor_scale"
							>
								<u>Natural Minor</u>
							</a>
							,&nbsp;
							<a
								className="body-text"
								href="https://en.wikipedia.org/wiki/Minor_scale#Harmonic_minor_scale"
							>
								<u>Harmonic Minor</u>
							</a>
							, and&nbsp;
							<a
								className="body-text"
								href="https://en.wikipedia.org/wiki/Minor_scale#Melodic_minor_scale"
							>
								<u>Melodic Minor</u>
							</a>
							.
						</p>
						<p className="body-text">
							Can't tell the difference between them? No worries! This audio
							classification program is trained to do it for you. Tip: The
							Scales is able to distinguish between these four different types
							of common scales being played.
						</p>

						<h2 className="body-text subtitle">
							<b>Step 1. Upload your .wav file</b>
						</h2>
						<p className="body-text unique">
							The program will use your .wav file and determine which scale (if
							any) is recorded in the file.
						</p>
						<br />

						<div className="container">
							<div className="row">
								<div className="offset-md-3 col-md-6">
									<div className="form-group files">
										<label>Upload Your File</label>
										<input
											type="file"
											class="form-control"
											multiple
											onChange={(event) => {
												//onChangeHandler fires when a file is uploaded
												this.onChangeHandler(event);
											}}
										/>
									</div>
									<button
										type="button"
										className="btn btn-success btn-block"
										onClick={(event) => {
											//onClickHandler fires when the upload button is clicked
											this.onClickHandler(event);
										}}
									>
										UPLOAD
									</button>
								</div>
							</div>
						</div>

						<h2 id="subtitle" className="body-text">
							<b>Step 2. Output!</b>
						</h2>
						<h3 className="body-text subtitle">
							<b>Your Scale Is:</b>
						</h3>
						<p className="body-text">
							<b>
								<u>{statement}</u>
							</b>
						</p>
						<p className="body-text unique" id="result-text">
							<div id="loader"></div>
						</p>
						<p className="body-text unique"></p>
						<br />
						<div className="container">
							<div className="row">
								<div className="offset-md-3 col-md-6">
									<button
										type="button"
										className="btn btn-success btn-block"
										id="spectro-button"
										onClick={(event) => {
											//showHideSpectrogram fires when the show spectrogram button is clicked
											this.showHideSpectrogram(event);
										}}
									>
										{this.state.button_title}
									</button>
								</div>
							</div>
						</div>

						<div className="saveme" id="spectro-container">
							<img id="input-image" src={spectroImage}></img>
						</div>
					</div>
				</body>
				<br />
			</div>
		);
	}
	//onChangeHandler fires when someone uploads a file
	onChangeHandler = (event) => {
		this.setState({
			selectedFile: event.target.files, //upon upload, state is updated
		});
	};
	//onClickHandler fires when a the upload button is pressed
	onClickHandler = (event) => {
		const data = new FormData();
		for (var x = 0; x < this.state.selectedFile.length; x++) {
			data.append("file", this.state.selectedFile[x]); //Request sent to server, file is appended to form data
		}
		console.log(data); //data is an object that contains the file (this is logged to the browser console)
		//axios is used to send AJAX requests, post takes in an endurl and form data (data)
		axios.post("http://localhost:8000/upload", data).then((res) => {
			console.log(res.statusText); //post is sent to the react app and the response is logged
		});
	};

	showHideSpectrogram = (event) => {
		if (this.state.button_title == "SHOW SCALE SPECTROGRAM") {
			this.setState({
				button_title: "HIDE SCALE SPECTROGRAM",
			});
			document.getElementById("spectro-container").style.display = "block"; //show the image
		} else {
			this.setState({
				button_title: "SHOW SCALE SPECTROGRAM",
			});
			document.getElementById("spectro-container").style.display = "none"; //hide the image
		}

		/* ---------------------------------------------------- */

		statement = `It is a Harmonic Minor`;
		console.log("It is a Major Scale");
	};
}

export default App;
