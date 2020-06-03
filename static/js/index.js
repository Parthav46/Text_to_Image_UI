t2i = () => {
	let loader = document.getElementById('loader')
	loader.style = "visibility: visible;"
	let str = document.getElementById('text').value;
	let csrf_token = document.getElementsByName('csrfmiddlewaretoken')[0].value
	console.log(csrf_token)
	$.post("./t2i", {
		csrfmiddlewaretoken: csrf_token,
		text: str
	},
	function(data, status) {
		loader.style = "visibility: hidden;"
		if(status === 'success') {
			document.getElementById('content').innerHTML = data;
		}
	});
}

rate = (rating) => {
	let id = document.getElementById('id').value
	let csrf_token = document.getElementsByName('csrfmiddlewaretoken')[0].value
	$.post("./rate", {
		csrfmiddlewaretoken: csrf_token,
		id: id, 
		rating: rating
	},
	function(data, status) {
		if(status === 'success') console.log('rating submitted');
		else console.log('failed to submit rating');
	});
}