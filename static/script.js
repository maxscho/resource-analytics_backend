const baseUrl = "http://localhost:9090"
// Upload
document.getElementById('fetchButton').addEventListener('click', function() {
    const fileInput = document.getElementById('fileInput');
    if (fileInput.files.length > 0) {
        document.getElementById('loader').style.display = "block";
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        fetch(`${baseUrl}/upload`, {
            method: 'POST',
            body: formData,
            credentials: 'include'
        })
        .then(response => response.json())
        .then(data => {
            // Handle the image
            const base64Image = data.image;
            document.getElementById('base64Image').src = 'data:image/jpeg;base64,' + base64Image;
                // Handle the table
            var logTable = new Tabulator("#dataTable", {data:data.table, autoColumns:true, layout:"fitColumns"});
            document.getElementById('analysisDropdown').disabled = false;
            document.getElementById('loader').style.display = "none";
        })
        .catch(error => console.error('Error:', error));
    } else {
        console.log("No file selected.");
    }
});

// Analysis
document.getElementById('analysisDropdown').addEventListener('change', function() {
    const basePath = baseUrl; //"http://localhost:9090/";
    const apiPath = this.value;
    if (apiPath) {
        document.getElementById('loader').style.display = "block";
        fetch(basePath + '/' + apiPath, {
            method: 'GET',
            credentials: 'include'
        })
        .then(response => {
            if (response.status === 401) {
                // Handle 401 Unauthorized
                return response.json().then(data => {
                    if (data.detail === "Session expired") {
                        alert("Session expired. The page will reload.");
                        window.location.reload();
                    }
                });
            } else {
                // Process response for other status codes
                return response.json();
            }
        })
        .then(data => {
            let content = '';
            if (data.image) {
                content += `<img src="data:image/jpeg;base64,${data.image}" style="max-width:100%; height:auto;">`;
            }
            if (data.table) {
                content += '<div id="resultTable"></div>'
            }
            if (data.text) {
                content += `<p>${data.text}</p>`;
            }
            document.getElementById('resultContainer').innerHTML = content;
            if (data.table) {
                var table = new Tabulator("#resultTable", {data:data.table, autoColumns:true, layout:"fitColumns"});
            }
        })
        .catch(error => console.error('Error:', error));
    }
    document.getElementById('loader').style.display = "none";
});
