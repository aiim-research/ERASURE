import jsonpickle
import json, sys

def compose(config):
    try:
        out_conf  = {}
        for item in config:
            if item.startswith('compose'):
                snippet = _get_snippet(config[item])
                for sub in snippet:
                    out_conf[sub]=_process_array(snippet[sub])#compose(snippet[sub])
            else:
                out_conf[item] = _process_array(config[item])

        return out_conf
    except BaseException:
        print("Incomplete/error configuration in:"+json.dumps(config, indent=2))
        raise


def _process_array(conf):
    try:
        if (isinstance(conf, list)):
            out_arr = []
            for arr_item in conf:
                if(isinstance(arr_item, dict)):
                    out_arr.append(compose(arr_item))
                else:
                    out_arr.append(arr_item)
            return out_arr
        return compose(conf) if isinstance(conf, dict) else conf
    except BaseException:
        print("Incomplete/error configuration in:"+json.dumps(conf, indent=2))
        raise 

def _get_snippet(snippet_path):
    # Read the config dictionary inside the config path
    with open(snippet_path, 'r') as config_reader:
        snippet = jsonpickle.decode(config_reader.read())        
    config_reader.close()

    return snippet

def propagate(config):
    try:
        if 'parameters' in config['experiment'] and 'propagate' in config['experiment']['parameters']:
            prop_list = config['experiment']['parameters']['propagate']
            for prop_item in prop_list:        
                for target in prop_item['in_sections']:
                    ord_secs = target.split('/')
                    main = ord_secs.pop(0)
                    for item in config[main]:
                        if 'parameters' not in item and len(ord_secs) == 0:
                            item['parameters']={}
                        for sub in ord_secs:
                            item=item[sub]
                        for key in prop_item['params']:
                            item['parameters'][key]=prop_item['params'][key]
        return config
    except BaseException:
        print("Incomplete/error configuration in:\n"+json.dumps(item, indent=2))
        raise 
