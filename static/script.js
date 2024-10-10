const baseUrl = "http://localhost:9090"
// Custom selector
document.querySelectorAll('.custom-select').forEach(function(select) {
    var container = document.createElement('div');
    container.className = 'custom-select-container';

    var display = document.createElement('div');
    display.className = 'custom-select-display';
    display.textContent = select.options[select.selectedIndex].textContent;
    container.appendChild(display);

    // Check if the original select is disabled
    if (select.disabled) {
        display.classList.add('disabled');
        display.removeEventListener('click', toggleDropdown); // Remove click event if disabled
    } else {
        display.addEventListener('click', toggleDropdown); // Ensure only enabled dropdowns are interactive
    }

    var optionsList = document.createElement('ul');
    optionsList.className = 'custom-select-options';
    Array.from(select.options).forEach(option => {
        var item = document.createElement('li');
        item.textContent = option.textContent;

        if (option.disabled) {
            item.classList.add('select-header'); // Add 'header' class for styling
            item.style.cursor = 'default'; // Ensure it doesn't look clickable
            //item.style.color = 'gray'; // Differentiate headers by color
            //item.style.backgroundColor = '#f0f0f0'; // Light background for headers
        } else {
            item.addEventListener('click', function() {
                select.value = option.value;
                display.textContent = option.textContent;
                optionsList.style.display = 'none';

                // Manually trigger the change event on the original select element
                if (typeof Event === 'function') {
                    var event = new Event('change');  // Modern browsers
                } else {
                    var event = document.createEvent('Event');  // For old browsers
                    event.initEvent('change', true, true);
                }
                select.dispatchEvent(event);
            });
        }
        optionsList.appendChild(item);
    });
    container.appendChild(optionsList);

//    display.addEventListener('click', function() {
//        optionsList.style.display = optionsList.style.display === 'none' ? 'block' : 'none';
//    });

    select.style.display = 'none';
    select.parentNode.insertBefore(container, select);
});
// Enabling and disabling custom selector
function toggleDropdown() {
    this.nextElementSibling.style.display = this.nextElementSibling.style.display === 'none' ? 'block' : 'none';
}
function setCustomSelectDisabled(select, isDisabled) {
    const container = select.parentNode.querySelector('.custom-select-display');
    if (isDisabled) {
        container.classList.add('disabled');
        container.removeEventListener('click', toggleDropdown);
    } else {
        container.classList.remove('disabled');
        container.addEventListener('click', toggleDropdown);
    }
}
// Click outside selector to close
document.addEventListener('click', function(e) {
    document.querySelectorAll('.custom-select-options').forEach(function(optionsList) {
        if (e.target.closest('.custom-select-container') !== optionsList.parentNode) {
            optionsList.style.display = 'none';
        }
    });
});
// Zoomable process model
document.getElementById('div2').addEventListener('wheel', function(event) {
    event.preventDefault();
    const img = document.getElementById('base64Image');

 // Retrieve or initialize scale
    let scale = img.dataset.scale ? parseFloat(img.dataset.scale) : 1;
    const scaleFactor = event.deltaY < 0 ? 1.1 : 0.9;
    scale *= scaleFactor;
    scale = Math.max(1, Math.min(scale, 10)); // Limit the scale between 1x and 10x
    img.dataset.scale = scale; // Store the scale in dataset for persistent state

    img.style.transform = `scale(${scale})`;
});

// Reset the zoom when the image src changes
document.getElementById('base64Image').addEventListener('load', function() {
    this.style.transform = 'scale(1)';
    this.dataset.scale = 1; // Reset scale
});

// Initial setup to make sure the image fills its container
window.addEventListener('load', function() {
    const img = document.getElementById('base64Image');
    img.style.height = '100%';
    img.dataset.scale = 1;
});


// Upload
document.getElementById('fetchButton').addEventListener('click', function() {
    const fileInput = document.getElementById('fileInput');
//    if (fileInput.files.length > 0) {
        document.getElementById('loader').style.display = "block";
        const formData = new FormData();
        if (fileInput.files.length > 0) {
            formData.append('file', fileInput.files[0]);
            url = `${baseUrl}/upload`;
        } else {
            const emptyBlob = new Blob([], { type: 'text/plain' });
            formData.append('file', emptyBlob,'empty.txt');
            url = `${baseUrl}/fake_upload`
        }
        fetch(url, {
            method: 'POST',
            body: formData,
            credentials: 'include'
        })
        .then(response => response.json())
        .then(data => {
            // Handle the image
            const base64Image = data.image;
            document.getElementById('base64Image').src = 'data:image/jpeg;base64,' + base64Image;
            // Save image
            localStorage.setItem('default_model', base64Image);
                // Handle the table
            var logTable = new Tabulator("#dataTable", {data:data.table, autoColumns:true, layout:"fitColumns"});
            //document.getElementById('analysisDropdown').disabled = false;
            setCustomSelectDisabled(document.getElementById('analysisDropdown'), false);
            document.getElementById('loader').style.display = "none";
        })
        .catch(error => console.error('Error:', error));
//    } else {
//        console.log("No file selected.");
//    }
});

// Analysis
const analysis = function() {
    const basePath = baseUrl; //"http://localhost:9090/";
    const apiPath = document.getElementById('analysisDropdown').value;
    if (apiPath) {
        document.getElementById('loader').style.display = "block";
        //const toggleState = document.getElementById('toggleSwitch').checked ? 'median' : 'average';
        fetch(basePath + '/' + apiPath, { // + `?mode=${toggleState}`, {
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
            content += '<p> ⓘ Plots and tables are interactive, hover over the plot for more information.</p>'  //'&#9432;'
            if (data.big_plot) {
                content += `<p>Scroll down for additional plots.</p>`;
            }
            if (data.image) {
                content += `<img src="data:image/jpeg;base64,${data.image}" style="max-width:100%; height:auto;">`;
            }
            if (data.plot) {
                content +=`<div id="plot"></div>`
            }
            if (data.table) {
                content += '<div id="resultTable"></div>'
            }
            if (data.text) {
                content += `<p>${data.text}</p>`;
            }
            if (data.big_plot) {
                content += `<div id="big_plot"></div>`;
            }
            if (data.process_model){
                const base64Image = data.process_model;
                document.getElementById('base64Image').src = 'data:image/jpeg;base64,' + base64Image;
            } else if(localStorage.getItem('default_model')){
                document.getElementById('base64Image').src = 'data:image/jpeg;base64,' + localStorage.getItem('default_model');
            }
            document.getElementById('resultContainer').innerHTML = content;
            if (data.table) {
                var table = new Tabulator("#resultTable", {data:data.table, autoColumns:true, layout:"fitColumns"});
            }
            if (data.plot) {
                figure = JSON.parse(data.plot)
                Plotly.newPlot('plot', figure.data, figure.layout)
            }
            if (data.big_plot) {
                big_figure = JSON.parse(data.big_plot)
                Plotly.newPlot('big_plot', big_figure.data, big_figure.layout)
            }
            document.getElementById('loader').style.display = "none";
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('loader').style.display = "none";
        });
    } else {
        document.getElementById('resultContainer').innerHTML = "";
    }
}

document.getElementById('analysisDropdown').addEventListener('change', analysis);
//document.getElementById('toggleSwitch').addEventListener('change', analysis);
document.addEventListener('DOMContentLoaded', (event) => {
    // Ensure the element exists before clicking
    const fetchButton = document.getElementById('fetchButton');
    if (fetchButton) {
        fetchButton.click(); // Simulate the click action
    }
});
