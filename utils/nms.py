import torch


def iou(x, y):
    x = x.view(-1, 4)
    y = y.view(-1, 4)
    a, b, c, d = x.split(1, 1)
    e, f, g, h = y.split(1, 1)
    s1 = (c - a) * (d - b)
    s2 = (g - e) * (h - f)
    x1 = torch.max(a, e)
    y1 = torch.max(b, f)
    x2 = torch.min(c, g)
    y2 = torch.min(d, h)
    w = (x2 - x1)
    w = torch.clamp(w, min=0)
    h = (y2 - y1)
    h = torch.clamp(h, min=0)
    s = w * h
    return s / (s1 + s2 - s + 1e-6)


def nms(det, threshold):
    _, ind = torch.sort(det[:, -1], descending=True)
    det = det[ind]
    # print('.......................')
    # print(det)
    for i in range(det.shape[0]):
        try:
            t = iou(det[i, 0:4], det[i + 1:, 0:4])
        except IndexError:
            break
        except ValueError:
            break
        # print(t)
        t = (t < threshold).float()
        # print(t)
        det[i + 1:, :] = det[i + 1:, :] * t
        ind = torch.nonzero(det[:, -1])
        # print(ind)
        det = det[ind].view(-1, 5)
        # print(det)
    # print(det)
    # print('********************')
    return det.view(-1, 5)
