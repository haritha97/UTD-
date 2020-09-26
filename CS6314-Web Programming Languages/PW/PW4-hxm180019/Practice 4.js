$(document).ready(function() {
	$.ajax({
		url: "movies.xml",
		datatype: "xml",
		success: function(data){
			$(data).find('movie').each(function(){
				var synopsis = $(this).find('synopsis').text();
				var score = $(this).find('score').text();
				var title = $(this).find('title').text();
				var director = $(this).find('director').text();
				var genre = $(this).find('genre');
				var genres='';
				$(genre).each(function(){
					genres += ' '  + $(this).text()+','
				})
				var person = $(this).find('person');
				var persons='';
				$(person).each(function(){
					persons += ' '  + $(this).attr('name')+','
				})
				
				var info = '<tr> <td>' + title + '</td>' + '<td>' + genres + '</td>' + '<td>' +director + '</td>' + '<td>' +persons+ '</td>' + '<td>' + synopsis + '</td>' + '<td>' + score +'</td> </tr>'
				$("table").append(info);
			});

		}
	});

	});