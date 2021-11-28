 const btn = document.querySelector('button');
 const getData= () => {};
 const sendData= () => {
	 
	axios.post('http://localhost:5000/test', {
    'name': 'Fred',
    'last_name': 'Flintstone'
  })
  .then(function (response) {
    console.log(response);
  })
  .catch(function (error) {
    console.log(error);
  });
	 
 };
 
 
 btn.addEventListener('click',sendData);
 