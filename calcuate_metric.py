def calcuate_metric(results, obj_list, logger, alpha = 0.2, args = None):

    table_ls = []
    auroc_sp_ls = []
    auroc_px_ls = []
    f1_sp_ls = []
    f1_px_ls = []
    aupro_px_ls = []
    aupro_sp_ls = []
    ap_sp_ls = []
    ap_px_ls = []
    iou_list = []
    iou_list_ls = []
    table_best_the = []

    names_list = np.array(results['cls_names'])
    gts_img = torch.tensor(results['gt_sp']).cpu().numpy()
    imgs_path = np.array(results['path'])

    anomaly_maps_raw = np.concatenate(results['anomaly_map_raw'], axis=0)
    anomaly_maps_new = np.concatenate(results['anomaly_map_new'], axis=0)
    gts_pixel = np.concatenate(results['imgs_masks'], axis=0)
    
    for obj in obj_list: 
        table = []
        table.append(obj)
        can_k = -2000

        object_index = np.where(names_list == obj)[0]
        img_path_list = imgs_path[object_index]
        pr_px_1 = anomaly_maps_raw[object_index,:,:].copy()
        pr_px_2 = anomaly_maps_new[object_index,:,:].copy()
        gt_sp = gts_img[object_index].copy()
        gt_px = gts_pixel[object_index,:,:].copy()
        pr_px =  normalize(gaussian_filter(alpha * pr_px_1 + (1 - alpha) * pr_px_2, sigma=8,axes = (1,2))) 
        pr_sp_1 = np.partition(pr_px_1.reshape(pr_px_1.shape[0],-1), kth=can_k)[:, can_k:]
        pr_sp_2 = np.partition(pr_px_2.reshape(pr_px_2.shape[0],-1), kth=can_k)[:, can_k:]

        pr_sp_tmp = np.mean(pr_sp_1, axis=1) + np.mean(pr_sp_2, axis=1)
        #pr_sp_tmp = np.max(pr_px, axis=(1,2))
        pr_sp_tmp = (pr_sp_tmp - pr_sp_tmp.min()) / (pr_sp_tmp.max() - pr_sp_tmp.min())
        pr_sp = pr_sp_tmp
        auroc_px = roc_auc_score(gt_px.ravel(), pr_px.ravel())    # pixel-level AUROC
        auroc_sp = roc_auc_score(gt_sp, pr_sp)  # image-level AUROC
        ap_sp = average_precision_score(gt_sp, pr_sp)  # image-level AP
        ap_px = average_precision_score(gt_px.ravel(), pr_px.ravel())  # pixel-level AP

        # f1_sp
        precisions, recalls, thresholds = precision_recall_curve(gt_sp, pr_sp)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        f1_sp = np.max(f1_scores[np.isfinite(f1_scores)])  # image-level F1-max

        # aupr
        aupro_sp = auc(recalls, precisions)  # image-level PRO
        #aupro_sp = 0
        
        # f1_px
        precisions, recalls, thresholds = precision_recall_curve(gt_px.ravel(), pr_px.ravel())
        f1_scores = (2 * precisions * recalls) / (precisions + recalls+ 1e-8)
        best_threshold = thresholds[np.argmax(f1_scores)]
        f1_px = np.max(f1_scores[np.isfinite(f1_scores)])   # pixel-level F1-max 
        iou = cal_iou(gt_px.ravel(), (pr_px.ravel()>best_threshold))  # mIoU
        iou_list.append(iou)
        print("{}--->  iou:{}   f1-max:{}  threshold:{}".format(obj,iou,f1_px,best_threshold))

        # aupro
        if len(gt_px.shape) == 4:
            gt_px = gt_px.squeeze(1)
        if len(pr_px.shape) == 4:
            pr_px = pr_px.squeeze(1)
        aupro_px = cal_pro_score(gt_px, pr_px) # pixel-level AUPRO
        #aupro_px = 0
        

        #----------------------------------start visualization --------------------------#
        print("visualization {}".format(obj))
        
        for i in range(len(img_path_list)):
            cls = img_path_list[i].split('/')[-2]
            filename = img_path_list[i].split('/')[-1]
            save_vis = os.path.join(args.save_path, 'imgs', obj, cls)
            vis_img = vis_img = cv2.resize(cv2.imread(img_path_list[i]), (args.image_size, args.image_size))
            visualization(save_root= save_vis, pic_name=filename, raw_image= vis_img, raw_anomaly_map= np.squeeze(pr_px[i]), raw_gt= np.squeeze(gt_px[i]), the = best_threshold)
        #----------------------------------end visualization --------------------------#

        table.append(str(np.round(auroc_px * 100, decimals=2)))   
        table.append(str(np.round(aupro_px * 100, decimals=2)))
        table.append(str(np.round(ap_px * 100, decimals=2)))

        table.append(str(np.round(f1_px * 100, decimals=2)))
        table.append(str(np.round(iou * 100, decimals=2)))


        table.append(str(np.round(auroc_sp * 100, decimals=2)))
        table.append(str(np.round(aupro_sp * 100, decimals=2)))

        table.append(str(np.round(ap_sp * 100, decimals=2)))
        table.append(str(np.round(f1_sp * 100, decimals=2)))
        table.append(str(np.round(best_threshold, decimals=3)))
        

        table_ls.append(table)
        auroc_sp_ls.append(auroc_sp)
        auroc_px_ls.append(auroc_px)
        f1_sp_ls.append(f1_sp)
        f1_px_ls.append(f1_px)
        aupro_px_ls.append(aupro_px)
        aupro_sp_ls.append(aupro_sp)
        ap_sp_ls.append(ap_sp)
        ap_px_ls.append(ap_px)
        iou_list_ls.append(iou)
        table_best_the.append(best_threshold)

    # logger
    table_ls.append(['mean', str(np.round(np.mean(auroc_px_ls) * 100, decimals=2)),
                     str(np.round(np.mean(aupro_px_ls) * 100, decimals=2)),
                      str(np.round(np.mean(ap_px_ls) * 100, decimals=2)),
                      str(np.round(np.mean(f1_px_ls) * 100, decimals=2)), 
                      str(np.round(np.mean(iou_list_ls) * 100, decimals=2)), 
                      str(np.round(np.mean(auroc_sp_ls) * 100, decimals=2)),
                      str(np.round(np.mean(aupro_sp_ls) * 100, decimals=2)),
                      str(np.round(np.mean(ap_sp_ls) * 100, decimals=2)),
                      str(np.round(np.mean(f1_sp_ls) * 100, decimals=2)),
                      str(np.round(np.mean(table_best_the), decimals=3))])
    results = tabulate(table_ls, headers=['objects', 'auroc_px', 'aupro_px', 'ap_px', 'f1_px', 'iou',"auroc_sp","aupro_sp","ap_sp", "f1_sp", "threshold"], tablefmt="pipe")
    logger.info("\n%s", results)
    print(args.checkpoint_path)
