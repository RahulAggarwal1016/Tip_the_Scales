var url = "https://teachablemachine.withgoogle.com/models/un_XbaxjZ/";



$(async function() {
    model = await tmImage.load(url + "model.json", url + "metadata.json")
    console.log("Model is loaded")

    maxPredictions = model.getTotalClasses();
    console.log(maxPredictions);

    $('#test-button').click(async function() {
        var prediction = await model.predict(document.getElementById("selected-image"));
        
    })
});

$.ajax({
    type: "POST",
    url: "../python/convert-to-jpg.py",
    data: { param: "filename.wav"}
  }).done(function( o ) {
     // print path to img
     console.log("pathtoimg");
  });



