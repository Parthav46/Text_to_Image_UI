$(document).ready(() => {
	$("#submit").on("click", () => {
		let str = $('#text').val();
		if (str === '' || str === null) return;
		$('#loader').css("visibility", "visible");
		let csrf_token = $("[name='csrfmiddlewaretoken']")[0].value;
		$.post("./t2i", {
			csrfmiddlewaretoken: csrf_token,
			text: str
		},
		(data, status) => {
			$('#loader').css("visibility", "hidden")
			if(status === 'success') {
				$('#content').html(data);
				$(".fa-star").on("click", (event) => {
					let t = event.target.id;
					rate(t, $('#id').val());
					$("#rating").val(t);
				});

				// star hover behavior
				$(".fa-star").on("mouseenter", (event) => {
					let t = event.target.id;
					for(let i = 1; i < 6; i++) {
						if(i <= t) $(".fa-star#"+i).addClass("checked");
						else $(".fa-star#"+i).removeClass("checked");
					}
				});
				$(".fa-star").on("mouseleave", (event) => {
					let t = $("#rating").val()
					for(let i = 1; i < 6; i++) {
						if(i <= t) $(".fa-star#"+i).addClass("checked");
						else $(".fa-star#"+i).removeClass("checked");
					}
				});
			}
		});
	});

	$("#random").on("click", () => {
		$.get("/randcap",
		(data, status) => {
			if(status === 'success') {
				$("#text").val(data);
			}
		}
		)
	});

	$(document).on('keypress', (event) => {
		if(event.keyCode === 13) {
			$("#submit").click();
		}
	})
})

rate = (rating, id) => {	
	let csrf_token = $("[name='csrfmiddlewaretoken']")[0].value;
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