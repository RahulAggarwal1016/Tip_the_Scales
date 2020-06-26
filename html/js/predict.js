var url = "https://teachablemachine.withgoogle.com/models/un_XbaxjZ/";

$(async function() {
    //loads model
    model = await tmImage.load(url + "model.json", url + "metadata.json")
    console.log("Model is loaded") 

    maxPredictions = model.getTotalClasses();

    //Uses the model to predict what the scale is
    document.getElementById("input-image").src = "./img/image.jpg";                             /* Rahul, you will need an <img> HTML tag to hold this image which is in the html/img directory. */
    var prediction = await model.predict(document.getElementById("input-image"));               /* Rahul, calls image that you saved in the <img> tag */
    console.log(prediction);

                    //Harmonic Minor               //Major                  //Melodic Minor             //Natural Minor             //Other
    var values = [prediction[0].probability, prediction[1].probability, prediction[2].probability, prediction[3].probability, prediction[4].probability];
    var maxProb = Math.max.apply(Math, values); //Finds which element has the highest value
    console.log(maxProb);

    //Determines which 
    if(maxProb == prediction[0].probability && prediction[0].probability >= 0.40) {
        console.log("It is Harmonic Minor");
        print_results("Harmonic Minor");
    } else if(maxProb == prediction[1].probability && prediction[1].probability >= 0.40) {
        console.log("It is Major");
        print_results("Major");
    } else if(maxProb == prediction[2].probability && prediction[2].probability >= 0.40) {
        console.log("It is Melodic Minor");
        print_results("Melodic Minor");
    } else if(maxProb == prediction[3].probability && prediction[3].probability >= 0.40) {
        console.log("It is Natural Minor");
        print_results("Natural Minor");
    } else {
        console.log("It is OTHEREERER");
        print_results("Other");
    }
});


function print_results(result) {
    //$('#result-text').html(result);                                                           /* Rahul, this prints the results on a <p> HTML tag. */
}


/*
$("#target").click(function(){
  $.ajax({
    type: "POST",
    url: "../python/convert-to-jpg.py",
    data: { param: "filename.wav"}
  }).done(function( o ) {
     // print path to img
     console.log("pathtoimg");
  });
});
*/