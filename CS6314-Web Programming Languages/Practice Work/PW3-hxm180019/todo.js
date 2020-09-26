$(document).ready(function()
        {
            $("ul").on("click","li",function(){
            $(this).toggleClass("done");
        });

        $("h3").on("click", "i", function(){
            $("#new").toggle();

        });

 		$("#new").keypress(function(event){
            if (event.which === 13)
            {
                $("ul").append("<li><span><i class='fa fa-trash'></i></span>"+ $("#new").val() + "</li>");
            }
 			
 		});


        $("ul").on("mouseenter", "li", function(){
            $(this).find("span").show();
        });
 		

        $("ul").on("mouseleave", "li", function(){
            $(this).find("span").hide();
        });
        
		$("ul").on("click", "span", function(){
			$(this).parent().remove();

 		});
        
        



 	  });