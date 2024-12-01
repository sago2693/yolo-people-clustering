document.addEventListener('DOMContentLoaded', () => {
    const clustersDiv = document.getElementById('clusters');
    let previewImg = null; // To hold the preview image element

    // Function to create a preview image on hover
    const showPreview = (imgSrc, event) => {
        if (!previewImg) {
            previewImg = document.createElement('img');
            previewImg.classList.add('preview-image');
            document.body.appendChild(previewImg);
        }
        previewImg.src = imgSrc;
        previewImg.style.left = `${event.pageX + 20}px`;
        previewImg.style.top = `${event.pageY + 20}px`;
        previewImg.style.display = 'block';
    };

    // Function to hide the preview image
    const hidePreview = () => {
        if (previewImg) {
            previewImg.style.display = 'none';
        }
    };

    // Load clusters from the server
    fetch('/get_clusters')
        .then(response => response.json())
        .then(clusters => {
            Object.keys(clusters).forEach(clusterId => {
                const clusterDiv = document.createElement('div');
                clusterDiv.classList.add('cluster');
                clusterDiv.dataset.clusterId = clusterId;

                // Cluster header with editable title
                const clusterHeader = document.createElement('div');
                clusterHeader.classList.add('cluster-header');

                const clusterTitle = document.createElement('input');
                clusterTitle.type = 'text';
                clusterTitle.value = `Cluster ${clusterId}`;
                clusterTitle.classList.add('editable-title');
                clusterTitle.addEventListener('change', () => {
                    clusterDiv.dataset.clusterId = clusterTitle.value; // Update dataset for the cluster
                });

                const deleteClusterBtn = document.createElement('button');
                deleteClusterBtn.classList.add('delete-cluster-btn');
                deleteClusterBtn.textContent = 'X';
                deleteClusterBtn.addEventListener('click', () => {
                    clusterDiv.remove(); // Remove the cluster from the DOM
                });

                clusterHeader.appendChild(clusterTitle);
                clusterHeader.appendChild(deleteClusterBtn);
                clusterDiv.appendChild(clusterHeader);

                const itemList = document.createElement('div');
                itemList.classList.add('item-list');

                clusters[clusterId].forEach(imageFilename => {
                    const imgDiv = document.createElement('div');
                    imgDiv.classList.add('thumbnail');
                    imgDiv.draggable = true;

                    // Add delete button to images
                    const deleteBtn = document.createElement('button');
                    deleteBtn.classList.add('delete-btn');
                    deleteBtn.textContent = 'X';
                    deleteBtn.addEventListener('click', () => {
                        imgDiv.remove(); // Remove the thumbnail from the DOM
                    });

                    const img = document.createElement('img');
                    img.src = `/static/thumbnails/${imageFilename}`;
                    img.alt = `Image ${imageFilename}`;
                    img.loading = "lazy"; // Enable lazy loading

                    // Hover to show expanded image
                    img.addEventListener('mouseenter', (event) => showPreview(`/static/images/${imageFilename}`, event));
                    img.addEventListener('mouseenter', (event) => showPreview(`/static/images/${imageFilename}`, event));
                    img.addEventListener('mouseleave', hidePreview);

                    imgDiv.appendChild(deleteBtn);
                    imgDiv.appendChild(img);
                    itemList.appendChild(imgDiv);
                });

                clusterDiv.appendChild(itemList);
                clustersDiv.appendChild(clusterDiv);

                // Enable drag-and-drop for this cluster
                new Sortable(itemList, {
                    group: 'clusters',
                    animation: 150,
                });
            });
        });

    // Save updated clusters
    document.getElementById('save').addEventListener('click', () => {
        const updatedClusters = {};
        document.querySelectorAll('.cluster').forEach(clusterDiv => {
            const clusterId = clusterDiv.querySelector('.editable-title').value;
            const images = Array.from(clusterDiv.querySelectorAll('.thumbnail img')).map(img => img.src.split('/').pop());
            updatedClusters[clusterId] = images;
        });

        fetch('/update_clusters', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(updatedClusters),
        }).then(response => response.json())
          .then(data => alert(data.message));
    });
});
