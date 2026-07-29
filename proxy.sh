cat << 'EOF' >> ~/.bashrc

# ===== Proxy helper =====
proxy() {
    local HOST="agent.baidu.com"
    local DEFAULT_PORT=8891

    case "$1" in
        off)
            unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
            echo "Proxy disabled"
            ;;
        status)
            if [ -n "$http_proxy" ]; then
                echo "Proxy enabled:"
                echo "  http_proxy=$http_proxy"
            else
                echo "Proxy disabled"
            fi
            ;;
        on|"")
            local PORT=$DEFAULT_PORT
            export http_proxy="http://${HOST}:${PORT}"
            export https_proxy="$http_proxy"
            export HTTP_PROXY="$http_proxy"
            export HTTPS_PROXY="$http_proxy"
            echo "Proxy enabled: ${HOST}:${PORT}"
            ;;
        *)
            local PORT=$1
            export http_proxy="http://${HOST}:${PORT}"
            export https_proxy="$http_proxy"
            export HTTP_PROXY="$http_proxy"
            export HTTPS_PROXY="$http_proxy"
            echo "Proxy enabled: ${HOST}:${PORT}"
            ;;
    esac
}
# ===== End Proxy helper =====

EOF

# 立刻生效
source ~/.bashrc