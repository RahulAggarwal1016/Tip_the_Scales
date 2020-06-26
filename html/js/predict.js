var url = "https://teachablemachine.withgoogle.com/models/un_XbaxjZ/";


$.ajax({
    type:"POST",
    url: "~/convert_to_jpg.py",
    data: {}
}).done(function(o) {
    console.log('YAY!');
})



model = await tmImage.load(url + "model.json"), url + "metadata.json");

maxPredictions = model.getTotalClasses();

