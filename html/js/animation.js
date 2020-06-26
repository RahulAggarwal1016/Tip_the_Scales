function showLoad() {
    // called when upload-button is clicked
    document.getElementById("result-text").textContent = "Processing audio...";
    document.getElementById("loader").style.display = "block";
}

function hideLoad() {
    // called when it's done (when data is received back from the server)

}

function showHideSpectrogram() {
    // if hidden, unhide it on click
    var spect_hidden = document.getElementById("spectrogram-button");
    if (spect_hidden.textContent == "SHOW SCALE SPECTROGRAM") {
        spect_hidden.textContent = "HIDE SCALE SPECTROGRAM";
        document.getElementById("spectro-container").style.display = "block";
        spect_hidden = False;
    }
    // if not hidden, hide it on click
    else {
        spect_hidden.textContent = "SHOW SCALE SPECTROGRAM";
        document.getElementById("spectro-container").style.display = "none";
        spect_hidden = True;
    }
}