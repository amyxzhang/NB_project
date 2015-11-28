var getCumulativeOffset = function(obj) {
    var left, top;
    left = top = 0;
    if (obj.offsetParent) {
        do {
            left += obj.offsetLeft;
            top  += obj.offsetTop;
        } while (obj = obj.offsetParent);
    }
    return {
        x : left,
        y : top
    };
};

var getTextPage = function() {
	text = "";
	$("#pf1 > div.pc").children().each(function () {
		     	  			
		ttt = this.innerText || this.textContent;
		
 	  	if ($.trim(ttt) == "") {
			return true;
		}
		
		ttt = ttt + '___l__' + $(this).css("left");
		
		if ($(this).css('color') != "rgb(35, 31, 32)") {
			ttt = ttt + '___c__' + $(this).css('color');
		}
		
		if ($(this).css("font-size") != "38px") {
			ttt = ttt + '___s__' + $(this).css("font-size");
		}
		
		text += '\n' + ttt;
	});
	return text;
};


var getText = function(x, y, w, h) {

	var x1 = x;
    var y1 = y;
    var h1 = h;
    var w1 = w;
    var b1 = y1 + h1;
    var r1 = x1 + w1;

	var text = "";
	
	var blah = $("#pf1 > div.pc").children()[0];
	
	if (blah.tagName == "IMG") {
		blah = $("#pf1 > div.pc").children()[1];
	}
	
	
	$("#pf1 > div.pc").children().each(function () {

		if (this.tagName != "IMG") {
			if ($(this).hasClass('c')) {
				return true;
			}

	      var x2 = $(this).offset().left;
	      var y2 = $(this).offset().top;
	      var h2 = this.offsetHeight;
	      var w2 = this.offsetWidth;
	      	if (w2 > 100) {
				w2 = w2/4.0;
			} 
			if (h2 > 20) {
				h2 = h2/2.3;
			} 

	      var b2 = y2 + h2;
	      var r2 = x2 + w2;
	      
	      

     	  if (!(b1 < y2 || y1 > b2 || r1 < x2 || x1 > r2)) {

     	  	if ((r2 - x1) > 15) {
     	  		
     	  		if ((r1 - x2) > 15) {
     	  			
     	  			if ((b2 - y1) > 5) {
     	  				
     	  				if ((b1 - y2) > 5) {
     	  		
     	  		if (!(x1 > x2 && y2 > 150 && y2 < 800 && x2 > 650 && x2 < 700 && w2 < 100 && w2 > 60 && h2 > 8 && h2 < 13)) {
     	  			

     	  		console.log(this);
     	  		if (!((this.innerText || this.textContent) == "CONCEPTS")) {
        			     	  			ttt = this.innerText || this.textContent;
     	  			
     	  			if ($.trim(ttt) == "") {
     	  				return true;
     	  			}
     	  			
     	  			ttt = ttt + '___l__' + $(this).css("left");
     	  			
     	  			if ($(this).css('color') != "rgb(35, 31, 32)") {
     	  				ttt = ttt + '___c__' + $(this).css('color');
     	  			}
     	  			
     	  			if ($(this).css("font-size") != "38px") {
     	  				ttt = ttt + '___s__' + $(this).css("font-size");
     	  			}
     	  			
        			text += '\n' + ttt;
        		
        		   	var d = document.createElement('div');
    	d.style.position = "absolute";
    	d.style.display = "block";
    	newx = x2;
		d.style.left = newx + 'px';
		newy = y2;
		d.style.top = newy + 'px';
			neww = w2;

		d.style.width= neww + 'px';

			newh = h2;
	
		d.style.height= newh + 'px';
		d.style.backgroundColor = 'red';
		d.style.opacity = "0.5";
    	document.getElementsByTagName('body')[0].appendChild(d);
        		}
        	}
        	}
        	}
        	}
        	}
        }

	   }
	});
	
	
	$(blah).children().each(function () {

		if (this.tagName != "IMG") {

	      var x2 = $(this).offset().left;
	      var y2 = $(this).offset().top;
	      var h2 = this.offsetHeight;
	      var w2 = this.offsetWidth;
	      	if (w2 > 100) {
				w2 = w2/4.0;
			} 
			if (h2 > 20) {
				h2 = h2/2.3;
			} 

	      var b2 = y2 + h2;
	      var r2 = x2 + w2;
	      
	      

     	  if (!(b1 < y2 || y1 > b2 || r1 < x2 || x1 > r2)) {

     	  	if ((r2 - x1) > 15) {
     	  		
     	  		if ((r1 - x2) > 15) {
     	  			
     	  			if ((b2 - y1) > 5) {
     	  				
     	  				if ((b1 - y2) > 5) {
     	  		
     	  		if (!(x1 > x2 && y2 > 150 && y2 < 800 && x2 > 650 && x2 < 700 && w2 < 100 && w2 > 60 && h2 > 8 && h2 < 13)) {
     	  			

     	  		console.log(this);
     	  		if (!((this.innerText || this.textContent) == "CONCEPTS")) {
     	  			ttt = this.innerText || this.textContent;
     	  			
     	  			if ($.trim(ttt) == "") {
     	  				return true;
     	  			}
     	  			
     	  			ttt = ttt + '___l__' + $(this).css("left");
     	  			
     	  			if ($(this).css('color') != "rgb(35, 31, 32)") {
     	  				ttt = ttt + '___c__' + $(this).css('color');
     	  			}
     	  			
     	  			if ($(this).css("font-size") != "38px") {
     	  				ttt = ttt + '___s__' + $(this).css("font-size");
     	  			}
     	  			
        			text += '\n' + ttt;
        		
        		   	var d = document.createElement('div');
    	d.style.position = "absolute";
    	d.style.display = "block";
    	newx = x2;
		d.style.left = newx + 'px';
		newy = y2;
		d.style.top = newy + 'px';
			neww = w2;

		d.style.width= neww + 'px';

			newh = h2;
	
		d.style.height= newh + 'px';
		d.style.backgroundColor = 'red';
		d.style.opacity = "0.5";
    	document.getElementsByTagName('body')[0].appendChild(d);
        		}
        	}
        	}
        	}
        	}
        	}
        }

	   }
	});
	return text;
	
};

window.onload = function() {

    var pdf_page = $("#pf1");
    
    tt = getTextPage(pdf_page);
    
    params = {'text': tt, 'page': window.location.href};
	$.post('/set_text_page', params, 
		function(res){
			console.log(res);
		}
	);

	console.log(tt);
    pdf_page = pdf_page.children()[0];
    
    offset = getCumulativeOffset(pdf_page);
    
    for (var i=0; i<coords.length; i++) {
    	coord = coords[i];
    	console.log(coord);
    	var d = document.createElement('div');
    	d.style.position = "absolute";
    	d.style.display = "block";
    	    	// 45 56
    	newx = offset.x + (coord.x/2.18) - 5;
		d.style.left = newx + 'px';
		newy = offset.y + (coord.y/2.21);
		d.style.top = newy + 'px';
		neww = coord.w/2.1;
		d.style.width= neww + 'px';
		newhh = coord.h/2.0;
		
		
		text = getText(newx, newy, neww, newhh);
		
		params = {"text": text, "page": window.location.href, "x": coord.x, "y": coord.y, "w": coord.w, "h": coord.h, "id": coord.id};
		
		$.post('/set_text', params, 
				function(res){
					console.log(res);
				}
			);
		
		console.log(text);
		d.style.height= newhh + 'px';
		d.style.backgroundColor = 'blue';
		d.style.opacity = "0.5";
    	document.getElementsByTagName('body')[0].appendChild(d);
		console.log(d);

    }

};