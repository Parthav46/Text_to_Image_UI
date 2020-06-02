t2i = () => {
	let str = document.getElementById('text').value;
	let csrf_token = document.getElementsByName('csrfmiddlewaretoken')[0].value
	console.log(csrf_token)
	$.post("./t2i", {
		csrfmiddlewaretoken: csrf_token,
		text: str
	},
	function(data, status) {
		if(status === 'success') {
			document.getElementById('content').innerHTML = data;
		}
	})
}