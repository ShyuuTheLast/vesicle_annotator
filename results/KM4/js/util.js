var getUrlParam = function(name){
    var rx = new RegExp('[\&|\?]'+name+'=([^\&\#]+)'),
    val = window.location.search.match(rx);
    return !val ? '':val[1];
}
