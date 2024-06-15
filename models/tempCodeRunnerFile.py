for i in range(len(cmc_results)):
        cmc_value = cmc_results[i]['cmcs'][0][0][0]
        
        # Extract label from filename
        eeg_filename = cmc_results[i]['eeg_filename'][0]  # Accessing the first element from the list
        filename = os.path.basename(eeg_filename)
        label = extract_label_from_filename(filename)
        
        # Append the CMC value to X and the label to y
        X.append(cmc_value)
        y.append(label)
        
        # Print debug information
        print(f'Processed file: {filename}, Label: {label}, CMC value: {cmc_value}')
    
    return np.array(X), np.array(y)